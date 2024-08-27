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
table_name = 'unegui_apartment'

unegui = 'https://www.unegui.mn/'


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
                    title VARCHAR(80),
                    description TEXT,
                    posted DATE,
                    floor VARCHAR(25),
                    balcony INTEGER,
                    year_of_completion INTEGER,
                    garage VARCHAR(20),
                    window VARCHAR(20),
                    number_of_apartment_floor INTEGER,
                    door VARCHAR(20),
                    area DOUBLE PRECISION,
                    which_floor INTEGER,
                    leasing VARCHAR(50),
                    number_of_windows INTEGER,
                    construction_progression VARCHAR(20),
                    price DOUBLE PRECISION,
                    type VARCHAR(50),
                    purpose VARCHAR(50),
                    location VARCHAR(100),
                    code VARCHAR(50)
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

def preprocess_data(df, room):
    if 'Байршил:' in df.columns and 'location:' in df.columns:
        df['Байршил:'] = df['Байршил:'] + '' + df['location:']
        df = df.drop(columns=['location:'])


    df = df.rename(columns={
        'Гарчиг': 'Title',
        'Дэлгэрэнгүй': 'Description',
        'Огноо': 'Date',
        'Шал:': 'Floor',
        'Тагт:': 'Balcony',
        'Ашиглалтанд орсон он:': 'Year of completion',
        'Гараж:': 'Garage',
        'Цонх:': 'Window',
        'Барилгын давхар:': 'Number of apartment floor',
        'Хаалга:': 'Door',
        'Талбай:': 'Area',
        'Хэдэн давхарт:': 'Which floor',
        'Лизингээр авах боломж:': 'Leasing',
        'Цонхны тоо:': 'Number of windows',
        'Барилгын явц:': 'Construction progression',
        'Үнэ': 'Price',
        'Төрөл:': 'Type',
        'Зориулалт:': 'Purpose',
        'Багтаамж:':'Capacity',
        'Байршил:':'Location',
        'Код:':'Code'
    })

    df = df.applymap(lambda x: x.replace('\n', '').strip() if isinstance(x, str) else x)
    df.fillna('')
    df.drop_duplicates(inplace=True)
    df['Rooms'] = room
    df = df[~df['Title'].str.contains('Бусад')]
    df = df[~df['Title'].str.contains('бусад')]
    df = df[~df['Price'].str.contains('Тэрбум')]
    
    df['Area'] = df['Area'].str.replace(' м²', '').astype(float)
    df['Price'] = df['Price'].str.split(',').str[0]
    df['Price'] = df['Price'].str.replace(' сая', '').astype(float)
    df['Balcony'] = df['Balcony'].str.extract(r'(\d+)', expand=False)
    df['id'] = df['id'].astype(int)

    current_date = datetime.now().strftime('%Y-%m-%d')
    yesterday_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    mongolian_to_date = {'Өнөөдөр': current_date, 'Өчигдөр': yesterday_date}

    df['posted'] = df['posted'].str.replace('Нийтэлсэн: ', '')
    df['posted'] = df['posted'].str.split(' ').str[0]
    df['posted'].replace(mongolian_to_date, inplace=True)
    df['posted'] = pd.to_datetime(df['posted'], errors='coerce')

    df.drop(columns=['Code'])
    df = df.fillna({'Floor': 'Unknown', 'Balcony': 0, 'Year of completion': '0', 'Garage': 'Unknown', 'Number of apartment floor': '0',
                    'Door': 'Area', 'Which floor': '0', 'Window': 'Unknown', 'Leasing': 'Unknown', 'Number of windows': '0', 'Construction progression': 'Unknown',
                        'Price': '0', 'Type': 'Unknown', 'Purpose': 'Unknown', 'Capacity': '0', 'Location': 'Unknown'})
    
    df[['Balcony', 'Year of completion', 'Number of apartment floor', 'Which floor', 'Number of windows']] = df[['Balcony', 'Year of completion',
                                                                        'Number of apartment floor', 'Which floor', 'Number of windows']].astype(int) 


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
    for room in range(1, 6):
        source_link = f'https://www.unegui.mn/l-hdlh/l-hdlh-zarna/oron-suuts-zarna/{room}-r/?page='
        for i in range(1, 220):
            link_1 = source_link + str(i)

            source = session.get(link_1).text
            soup = BeautifulSoup(source, 'lxml')

            list_of_apart = soup.find('div', class_='list-announcement-assortiments')
            if list_of_apart is None:
                logging.warning("No list of apartments found on page %d", i)
                continue

            for block in list_of_apart.find_all('div', class_='advert'):
                link = block.find("a", class_='swiper-slide')['href'].split("?")[0]
                links.append(link)

        logging.info("Links fetched: %d links", len(links))

        detail_list = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = executor.map(get_data, links)

        for result in results:
            if result is not None:
                detail_list.append(result)

        logging.info("Data fetched: %d records", len(detail_list))

        if detail_list:
            df = pd.DataFrame(detail_list)
            processed_df = preprocess_data(df, room)

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