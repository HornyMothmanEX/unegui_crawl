import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import requests
import re
import pandas as pd
import asyncio
from playwright.sync_api import sync_playwright

def get_links(link):
    with httpx.Client() as client:
        response = client.get(link)
        soup = BeautifulSoup(response.text, 'html.parser')
        urls = []

        main = soup.find('div', class_='container p-0')

        if main:
            for a_tag in main.find_all('a', href=True):
                full_url = urljoin(link, a_tag['href'])
                urls.append(full_url)

    return urls

def product_links(link):
    with httpx.Client() as client:
        response = client.get(link)
        soup = BeautifulSoup(response.text, 'html.parser')
        urls = []
        
        # Find all "viewMore" sections on the page
        view_more_sections = soup.find_all('div', class_='viewMore')
        
        for section in view_more_sections:
            # Extract links from each "viewMore" section
            for a_tag in section.find_all('a', href=True):
                full_url = urljoin(link, a_tag['href'])
                urls.append(full_url)
                
    return urls

def get_description(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)

        # Wait for the element to be available
        page.wait_for_selector('div.content', timeout=20000)  # Wait up to 60 seconds
        element = page.locator('div.product-descriptiondetails > .content')
        
        # Extract and return text from the specific element
        text = element.inner_text()
        browser.close()
        return text


def find_other_ingredients(soup):
    p_tags = soup.find_all('p')
    extracted_text = []
    in_target_section = False

    for p in p_tags:
        text = p.get_text(strip=True)
        if 'Other Ingredients' in text:
            in_target_section = True
            # Remove "Other Ingredients" label and add the following text
            extracted_text.append(text.replace("Other Ingredients", "").strip())
        elif 'Ingredients' in text:
            in_target_section = True
            # Remove "Other Ingredients" label and add the following text
            extracted_text.append(text.replace("Ingredients:", "").strip())
        elif 'Precaution' in text or 'Storage' in text:
            # Stop extraction once "Storage" is found
            break
        elif in_target_section:
            extracted_text.append(text)


    cleaned_text = ' '.join(extracted_text).replace('&nbsp;', ' ').strip().lower()
    return cleaned_text

def get_best_for(soup, label):
    # Find the element containing the label
    label_element = soup.find(string=lambda text: text and label in text)
    
    if label_element:
        # Find the parent of the label element to start navigating from
        parent = label_element.find_parent()
        
        # Iterate through siblings of the parent
        for sibling in parent.find_all_next():
            if sibling.name == 'ul':
                # Capture and return the concatenated text from all list items in the unordered list
                return ' '.join(li.get_text(strip=True) for li in sibling.find_all('li'))
    
    return ''



def get_text_after_label(soup, label):
    # Find the element containing the label
    label_element = soup.find(string=lambda text: text and label in text)
    
    if label_element:
        # Find the parent of the label element to start navigating from
        parent = label_element.find_parent()
        
        # Iterate through siblings of the parent
        for sibling in parent.find_all_next():
            if sibling.name == 'p':
                # Return text from the first relevant sibling found
                return sibling.get_text(strip=True)
            elif sibling.name == 'hr':
                # Stop if an 'hr' tag is encountered (end of section)
                break
    
    return ''

def extract_manufactured_nation(soup):
    manufactured_in_text = ""
    p_tags = soup.find_all('p')

    for p in p_tags:
        text = p.get_text(strip=True)
        if 'Manufactured in' in text:
            # Extract everything after "Manufactured in"
            manufactured_in_text = text.split('Manufactured in')[-1].strip()
            break

    return manufactured_in_text if manufactured_in_text else ''

with open('links.txt', 'r') as file:
    # Read all lines from the file
    lines = file.readlines()
all_links = []
# Process and print each line
for line in lines:
    all_links.append(line.strip())

all_details = []  # List to store details of each product

try:
    with httpx.Client() as client:
        for i, link in enumerate(all_links):  # Loop through the first 10 links
            source = client.get(link)
            soup = BeautifulSoup(source.text, 'html.parser')
            detail = {}

            content = soup.find('div', class_='fade show')

            try:
                id_num = content.find('span', class_='product-id text--bold')
                detail['id'] = id_num.text if id_num else None
            except Exception as e:
                print(f"Error finding product ID for link {i + 1}: {e}/n {all_links[i]}")

            try:
                name = soup.find('div', class_='d-none d-sm-block').h1.text
                detail['product_name'] = name
            except Exception as e:
                print(f"Error finding product name for link {i + 1}: {e}/n {all_links[i]}")

            try:
                brand = soup.find('div', class_='d-none d-sm-block').a.text
                detail['brand'] = brand
            except Exception as e:
                print(f"Error finding brand for link {i + 1}: {e}/n {all_links[i]}")

            try:
                breadcrumb = soup.find('ol', class_='breadcrumb text-size--small col', itemprop='category')
                model = breadcrumb.find_all('li', class_='breadcrumb-element')
                product_type = re.sub(r'\s+', ' ', model[-2].text.strip())
                detail['type'] = product_type
            except Exception as e:
                print(f"Error finding product type for link {i + 1}: {e}/n {all_links[i]}")

            try:
                price = soup.find('span', itemprop='price', class_='d-none').text
                detail['usual_price'] = price
            except Exception as e:
                print(f"Error finding usual price for link {i + 1}: {e}/n {all_links[i]}")

            try:
                vip_price = soup.find('span', itemprop='sale_price', class_='d-none').text
                detail['vip_price'] = vip_price
            except Exception as e:
                print(f"Error finding VIP price for link {i + 1}: {e}/n {all_links[i]}")

            divs1 = soup.find_all('div', class_='col-sm-12 col-md-4 col-lg-2 d-flex d-md-block justify-content-between pb-2')

            for div in divs1:
                try:
                    label = div.find('span', class_='d-block product-overview--label')
                    if label and label.text == "Form":
                        bold_span = div.find('span', class_='text--bold')
                        if bold_span and bold_span.a:
                            form = bold_span.a.text.strip()
                            detail['form'] = form
                except Exception as e:
                    print(f"Error finding form for link {i + 1}: {e}/n {all_links[i]}")

            
            try:
                servings = soup.find('div', class_='product-uom d-none').text
                detail['servings_per_container'] = servings
            except Exception as e:
                print(f"Error finding servings per container for link {i + 1}: {e}/n {all_links[i]}")

            try:
                description = get_description(all_links[i])
                detail['description'] = description
            except Exception as e:
                print(f"Error finding description for link {i + 1}: {e}/n {all_links[i]}")
            
            try:
                usage = soup.find('div', id='product-usagedirection').text
                detail['usage_direction'] = usage
            except Exception as e:
                print(f"Error finding usage direction for link {i + 1}: {e}/n {all_links[i]}")

            try:
                key_ingredient_list = soup.find_all('div', class_='ingredient')
                key_ingredient = [div.find('a').text.strip() for div in key_ingredient_list]
                # Join the list into a single string, separating each ingredient with a comma
                key_ingredient = ', '.join(key_ingredient)
                key_ingredient = re.sub(r'\s+', ' ', key_ingredient.strip())
                detail['key_ingredient'] = key_ingredient
            except Exception as e:
                print(f"Error finding key ingredient for link {i + 1}: {e}/n {all_links[i]}")

            try:
                other_ingredient_section = soup.find('div', id='product-supplements')
                other_ingredients = find_other_ingredients(other_ingredient_section)
                detail['other_ingredients'] = other_ingredients
            except Exception as e:
                print(f"Error finding other ingredients for link {i + 1}: {e}/n {all_links[i]}")

           

            try:
                detail['storage'] = get_text_after_label(soup, "Storage")
                detail['precaution'] = get_text_after_label(soup, "Precaution")
                detail['best_for'] = get_best_for(soup, 'Best for people with')
                detail['manufacturer'] = extract_manufactured_nation(soup)
            except Exception as e:
                print(f"Error finding additional details for link {i + 1}: {e}/n {all_links[i]}")

            try:
                usp_list = soup.find_all('div', class_='usp-list-label')
                usp_texts = [div.find('span').text.strip() for div in usp_list]
                joined_usp_texts = ', '.join(usp_texts)
                detail['feature'] = joined_usp_texts
            except Exception as e:
                print(f"Error finding USP texts for link {i + 1}: {e}/n {all_links[i]}")
            
            try:
                detail['link'] = all_links[i]
            except:
                print(f"Error finding link {i + 1}: {e}/n {all_links[i]}")
            all_details.append(detail)  # Add the detail dictionary to the list

except Exception as e:
    print(f"General error occurred: {e}")

df = pd.DataFrame(all_details)


df.to_csv('product_details.csv', index=False)
print("Data has been written to 'product_details.csv'")

