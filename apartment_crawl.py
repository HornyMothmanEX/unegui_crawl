from bs4 import BeautifulSoup
import requests
import pandas as pd
import concurrent.futures
import re
import datetime
from datetime import timedelta, datetime
import numpy as np

unegui = 'https://www.unegui.mn/'


dataf = []

for room in range(1, 2):
    source_link = f'https://www.unegui.mn/l-hdlh/l-hdlh-zarna/oron-suuts-zarna/{room}-r/?page='
    detail_list = []
    def get_data(link):
        source_1 = requests.get(unegui+link).text
        soup_1 = BeautifulSoup(source_1, 'lxml')
        content = soup_1.find('div', class_ = 'announcement-content-container')

        detail_dic = {}
        
        id_sec = content.find('div', class_='announcement__details')
        id = id_sec.find('span', {'itemprop': 'sku'})
        detail_dic['Зарын дугаар'] = id

        header = content.find('div', class_='announcement-content-header').h1.text
        detail_dic['Гарчиг'] = header

        loc = content.find('div', class_='announcement-meta__left').span.text
        detail_dic['Байршил'] = loc

        description = content.find('div', class_='announcement-description').p.text

        detail_dic['Дэлгэрэнгүй'] = description

        datet = content.find('div', class_='announcement__details').span.text

        detail_dic['Огноо'] = datet

        details = content.find('div', class_='announcement-characteristics clearfix')

        for ul in details.find_all('li'):
            has_a = ul.find('a')
            if has_a is None:
                detail_key = ul.find('span', class_='key-chars').text
                detail_value = ul.find('span', class_='value-chars').text
            else:
                detail_key = ul.find('span', class_='key-chars').text
                detail_value = has_a.string

            detail_dic[detail_key] = detail_value


        price_content = soup_1.find('div', class_ = 'announcement-price__cost')
        new_price = price_content.contents[4]

        detail_dic['Үнэ'] = new_price
        detail_list.append(detail_dic)

    def find_last_page_number(source_link):
        last_page_url = source_link + "9999999"
        response = requests.get(last_page_url)

        if response.status_code == 200:
            last_page_number = re.search(r'page=(\d+)', response.url)
            if last_page_number:
                return int(last_page_number.group(1))
        return 1

    last_page_number = find_last_page_number(source_link)

    links = []
    for i in range(1, last_page_number):


        link_1 = source_link + str(i)

        source = requests.get(link_1).text
        soup = BeautifulSoup(source, 'lxml')

        # list_of_car = soup.find('ul', class_ = 'list-simple__output')
        list_of_apart = soup.find('div', class_ = 'list-announcement-assortiments')

        for block in list_of_apart.find_all('div', class_='advert'):
            link = block.find("a", class_='swiper-slide')['href'].split("?")[0]
            links.append(link)

    def extract_datetime(date_str):
        now = datetime.now()
        today = now.date()

        if 'Өчигдөр' in date_str:
            if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                # If full date is provided, use it instead of calculating yesterday
                date_part = date_str.split(' ')[0]
            else:
                date_part = today - timedelta(days=1)  # Yesterday
            time_part = date_str.split(' ')[-1]  # Get the last part as time
            datetime_str = f"{date_part} {time_part}"
        
        elif 'Өнөөдөр' in date_str:
            if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                # If full date is provided, use it instead of today's date
                date_part = date_str.split(' ')[0]
            else:
                date_part = today  # Today
            time_part = date_str.split(' ')[-1]  # Get the last part as time
            datetime_str = f"{date_part} {time_part}"
        
        elif 'Нийтэлсэн:' in date_str:
            datetime_str = date_str.split(': ')[1].strip()
        
        elif re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}', date_str):
            datetime_str = date_str.strip()
        
        else:
            # If the format doesn't match known patterns, log or raise an error
            raise ValueError(f"Unknown date format: {date_str}")

        # Convert to datetime object
        try:
            return datetime.strptime(datetime_str, '%Y-%m-%d %H:%M')
        except ValueError:
            raise ValueError(f"Invalid date format: {datetime_str}")
    def convert_price(price):
        if 'Тэрбум' in price:
            price = price.replace('Тэрбум', '').strip()
            if price:
                return float(price) * 1000  # 1 Тэрбум = 1000 сая
            else:
                return 1000  # If 'Тэрбум' alone, assume 1 billion (1000 сая)
        elif 'сая' in price:
            return float(price.replace('сая', '').strip())
        else:
            return 0

    print('next')    

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(get_data, links)


    df = pd.DataFrame(detail_list)
    if 'Байршил:' in df.columns:
        df = df.drop(columns=['Байршил:'])
    if 'location:' in df.columns:
        df = df.drop(columns=['location:'])
    if 'Код:' in df.columns:
        df = df.drop(columns=['Код:'])
    df = df.rename(columns={
        'Гарчиг': 'title',
        'Зарын дугаар': 'id',
        'Байршил': 'location',
        'Дэлгэрэнгүй': 'description',
        'Огноо': 'date',
        'Шал:': 'floor',
        'Тагт:': 'balcony',
        'Ашиглалтанд орсон он:': 'year_of_completion',
        'Гараж:': 'garage',
        'Цонх:': 'window_type',
        'Барилгын давхар:': 'number_of_apartment_floor',
        'Хаалга:': 'door',
        'Талбай:': 'area',
        'Хэдэн давхарт:': 'which_floor',
        'Лизингээр авах боломж:': 'leasing',
        'Цонхны тоо:': 'number_of_windows',
        'Барилгын явц:': 'construction_progression',
        'Үнэ': 'price',
        'Төрөл:': 'type',
        'Зориулалт:': 'purpose',
        'Багтаамж:':'capacity'
    })



    df = df.map(lambda x: x.replace('\n', '').strip() if isinstance(x, str) else x)
    df['price'] = df['price'].apply(convert_price)
    df['date'] = df['date'].apply(extract_datetime)
    df.fillna('')

    df['rooms'] = room


    df = df[~df['title'].str.contains('Бусад')]
    df = df[~df['title'].str.contains('бусад')]

    df['area'] = df['area'].str.replace(' м²', '').astype(float)
    df['balcony'] = df['balcony'].str.extract(r'(\d+)', expand=False)

    # df = df.drop(['Capacity','Code'], axis=1)
    df = df.fillna({'floor': 'unknown', 'balcony': 0, 'year_of_completion': '0', 'garage': 'unknown', 'number_of_apartment_floor': '0',
                    'door':'unknown', 'area': 0, 'which_floor': '0', 'window_type': 'unknown', 'leasing': 'unknown', 'number_of_windows': '0', 
                    'construction_progression': 'unknown', 'price': '0'})
  
    df[['balcony', 'year_of_completion', 'number_of_apartment_floor', 'which_floor', 'number_of_windows']] = df[['balcony', 'year_of_completion',
                                                                        'number_of_apartment_floor', 'which_floor', 'number_of_windows']].astype(int) 
    

    

    dataf.append(df)

dataf = pd.concat(dataf, ignore_index=True)
current_time = datetime.now()

unique_filename = current_time.strftime("%Y%m%d_%H%M%S") + '_unegui_apartment.csv'
dataf.to_csv(unique_filename, index=False, encoding='utf-8')
print('done')