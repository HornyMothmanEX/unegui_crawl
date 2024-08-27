import logging
from bs4 import BeautifulSoup
import requests
import pandas as pd
import concurrent.futures
from datetime import datetime, timedelta
import psycopg2
import schedule
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
hostname = '172.16.116.195'
database = 'postgres'
username = 'postgres'
password = 'RVs8bgUzH'
port_id = 5432
schema_name = 'nes_aid'
table_name = 'unegui_cars'

unegui = 'https://www.unegui.mn/'
source_link = 'https://www.unegui.mn/avto-mashin/-avtomashin-zarna/?page='

# Create a session for retries
session = requests.Session()
retry = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)
session.mount("http://", adapter)

def create_table():
    conn = psycopg2.connect(
        host=hostname,
        dbname=database,
        user=username,
        password=password,
        port=port_id
    )
    cur = conn.cursor()

    try:
        create_script = f"""CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} (
                    id SERIAL PRIMARY KEY,
                    title VARCHAR(50),
                    "description" TEXT,
                    posted DATE,
                    "engine_capacity" DOUBLE PRECISION,
                    gearbox VARCHAR(50),
                    "wheel_position" VARCHAR(50),
                    type VARCHAR(50),
                    color VARCHAR(50),
                    "year_of_manufacture" INTEGER,
                    "imported_year" INTEGER,
                    engine VARCHAR(50),
                    "saloon_color" VARCHAR(50),
                    leasing VARCHAR(50),
                    transmission VARCHAR(50),
                    "road_traveled" BIGINT,
                    condition VARCHAR(50),
                    doors INTEGER,
                    price DOUBLE PRECISION,
                    manufacturer VARCHAR(50),
                    mark VARCHAR(50)
                );"""

        cur.execute(create_script)
        conn.commit()
    except Exception as error:
        logging.error("Error creating table: %s", error)
        conn.rollback()
    finally:
        cur.close()
        conn.close()

def import_data_to_postgres(df):
    conn = psycopg2.connect(
        host=hostname,
        dbname=database,
        user=username,
        password=password,
        port=port_id
    )
    cur = conn.cursor()

    try:
        data = [tuple(row) for row in df.values]

        placeholders = ','.join(['%s'] * len(df.columns))

        sql = f"INSERT INTO {schema_name}.{table_name} VALUES ({placeholders}) ON CONFLICT (id) DO NOTHING"

        cur.executemany(sql, data)

        conn.commit()
    except Exception as error:
        logging.error("Error importing data: %s", error)
        conn.rollback()
    finally:
        cur.close()
        conn.close()

def preprocess_data(df):
    df = df.rename(columns={
        'Мотор багтаамж:': 'engine_capacity',
        'Хурдны хайрцаг:': 'gearbox',
        'Хүрд:': 'wheel_position',
        'Төрөл:': 'type',
        'Өнгө:': 'color',
        'Үйлдвэрлэсэн он:': 'year_of_manufacture',
        'Орж ирсэн он:': 'imported_year',
        'Хөдөлгүүр:': 'engine',
        'Дотор өнгө:': 'saloon_color',
        'Лизинг:': 'leasing',
        'Хөтлөгч:': 'transmission',
        'Явсан:': 'road_traveled',
        'Нөхцөл:': 'condition',
        'Хаалга:': 'doors',
        'Үнэ': 'price'
    })

    df = df.applymap(lambda x: x.replace('\n', '').strip() if isinstance(x, str) else x)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    df['title'] = df['title'].str.split(',').str[0]
    df = df[~df['title'].str.contains('Бусад')]
    df = df[~df['title'].str.contains('бусад')]
    df['engine_capacity'] = df['engine_capacity'].replace('Цахилгаан', '0')
    df['engine_capacity'] = df['engine_capacity'].str.replace(' л', '').astype(float)
    
    # Clean the 'road_traveled' column
    df['road_traveled'] = df['road_traveled'].str.replace(r'\D', '', regex=True).apply(lambda x: int(x) if x.isdigit() else 0)
    
    df['year_of_manufacture'] = df['year_of_manufacture'].astype(int)
    df['imported_year'] = df['imported_year'].astype(int)
    df['doors'] = df['doors'].astype(int)
    df['id'] = df['id'].astype(int)

    current_date = datetime.now().strftime('%Y-%m-%d')
    yesterday_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    mongolian_to_date = {'Өнөөдөр': current_date, 'Өчигдөр': yesterday_date}

    df['posted'] = df['posted'].str.replace('Нийтэлсэн: ', '')
    df['posted'] = df['posted'].str.split(' ').str[0]
    df['posted'].replace(mongolian_to_date, inplace=True)
    df['posted'] = pd.to_datetime(df['posted'], errors='coerce')

    df[['manufacturer', 'mark']] = df['title'].str.split(n=1, expand=True)
    df = df[~df['price'].str.contains('Тэрбум')]
    df['price'] = df['price'].str.replace(' сая', '').astype(float)

    Q1 = df['road_traveled'].quantile(0.25)
    Q3 = df['road_traveled'].quantile(0.75)
    IQR = Q3 - Q1

    outlier_threshold = 1.5

    outlier_mask = (df['road_traveled'] < Q1 - outlier_threshold * IQR) | (
                df['road_traveled'] > Q3 + outlier_threshold * IQR)

    df = df[~outlier_mask]

    return df

def get_data(link):
    try:
        source_1 = session.get(unegui + link).text
        soup_1 = BeautifulSoup(source_1, 'lxml')
        content = soup_1.find('div', class_='announcement-content-container')

        if content is None:
            logging.warning("Content not found for link: %s", link)
            return None

        detail_dic = {}

        idt = content.find('span', class_='number-announcement')
        if idt and idt.span:
            detail_dic['id'] = idt.span.text
        else:
            logging.warning("ID not found for link: %s", link)
            return None

        header = content.find('div', class_='announcement-content-header')
        if header and header.h1:
            detail_dic['title'] = header.h1.text
        else:
            logging.warning("Title not found for link: %s", link)
            return None

        description = content.find('div', class_='announcement-description')
        if description and description.p:
            detail_dic['description'] = description.p.text
        else:
            logging.warning("Description not found for link: %s", link)
            return None

        datet = content.find('div', class_='announcement__details')
        if datet and datet.span:
            detail_dic['posted'] = datet.span.text
        else:
            logging.warning("Posted date not found for link: %s", link)
            return None

        details = content.find('div', class_='announcement-characteristics clearfix')
        if details:
            for ul in details.find_all('li'):
                has_a = ul.find('a')
                if has_a is None:
                    detail_key = ul.find('span', class_='key-chars').text
                    detail_value = ul.find('span', class_='value-chars').text
                else:
                    detail_key = ul.find('span', class_='key-chars').text
                    detail_value = has_a.string
                detail_dic[detail_key] = detail_value

        price_content = soup_1.find('div', class_='announcement-price__cost')
        if price_content and len(price_content.contents) > 4:
            new_price = price_content.contents[4]
            detail_dic['Үнэ'] = new_price
        else:
            logging.warning("Price not found for link: %s", link)
            return None

        logging.info("Data fetched for link: %s", link)
        return detail_dic

    except Exception as e:
        logging.error("Error fetching data for link %s: %s", link, e)
        return None

def job():
    logging.info("Job started")

    create_table()

    logging.info("Table checked")

    links = []
    for i in range(1, 220):
        link_1 = source_link + str(i)

        source = session.get(link_1).text
        soup = BeautifulSoup(source, 'lxml')

        list_of_car = soup.find('div', class_='list-announcement-assortiments')
        if list_of_car is None:
            logging.warning("No list of cars found on page %d", i)
            continue

        for block in list_of_car.find_all('div', class_='advert'):
            link = block.find("a", class_='swiper-slide')['href'].split("?")[0]
            links.append(link)

    logging.info("Links fetched: %d links", len(links))

    detail_list = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(get_data, links)

    for result in results:
        if result is not None:
            detail_list.append(result)

    logging.info("Data fetched: %d records", len(detail_list))

    if detail_list:
        df = pd.DataFrame(detail_list)
        processed_df = preprocess_data(df)

        logging.info("Data preprocessed")

        import_data_to_postgres(processed_df)

        logging.info("Data imported")
    else:
        logging.info("No data to process and import")
    
    del df
    del processed_df
    del detail_list
    del links
    gc.collect()  # Force garbage collection to release memory

    logging.info("Job ended")

schedule.every().day.at("01:08").do(job)

while True:
    schedule.run_pending()

    hour = datetime.now().strftime("%H:%M:%S")
    logging.info('Working %s', hour)
    time.sleep(10)