import os
import json
import time
import requests
from random import randint
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

from JungoNara import JungoNara
from src.dbControl.connect_db import connectDB
from src.dbControl.close_connection import close_connection
from src.dbControl.insert_product import insert_product
from utils.URLCache import URLCache
import utils.utils as utils
import utils.imageToS3 as imageToS3
import utils.thecheatapi as thecheatapi
from src.data_processing import find_phone_num

class Cellphone(JungoNara):
    def __init__(self, base_url, bucket_name, delay_time=None, saving_html=False):
        super().__init__(delay_time, saving_html)
        self.base_url = base_url
        self.jungo_url = "https://cafe.naver.com"
        self.bucket_name = bucket_name
        
        load_dotenv()
        self.api_url = os.getenv('THE_CHEAT_URL')
        self.api_key = os.getenv('X-TheCheat-ApiKey')
        self.enc_key = os.getenv('ENC_KEY')
        
        options = webdriver.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument('--headless')
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        self.driver = webdriver.Chrome(options=options)

    def _dynamic_crawl(self, url: str):
        assert url.startswith(self.jungo_url), "Given url does not seem to be from cellphone category."
        
        self.driver.get(url)
        time.sleep(self._calculate_delay())
        
        try:
            wait = WebDriverWait(self.driver, 10)
            iframe = wait.until(EC.presence_of_element_located((By.ID, "cafe_main")))
            self.driver.switch_to.frame(iframe)
        except Exception as e:
            print(e)
            return
        
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        description_text = self._extract_description(soup)
        phone_num, is_tell = self._extract_phone_numbers(soup, description_text)
        if phone_num is None:
            return
        is_fraud = self._check_fraud(phone_num)
        
        product_info = self._extract_product_info(soup)
        product_info.update({
            "phone_num": phone_num,
            "is_fraud": is_fraud,
            "is_find": is_tell is False and phone_num is not None
        })
        
        self._save_to_db(product_info)
        self._save_images(soup, product_info["product_id"])
        
        self.driver.switch_to.default_content()

    def _calculate_delay(self):
        if isinstance(self.delay_time, float):
            return self.delay_time
        elif isinstance(self.delay_time, tuple):
            return float(randint(self.delay_time[0], self.delay_time[1]))
        return 0
    
    def _extract_description(self, soup):
        se_module = soup.find_all('div', class_="se-section se-section-text se-l-default")
        return "\n".join([span.get_text(strip=True) for module in se_module for span in module.select('div > p > span')])

    def _extract_phone_numbers(self, soup, description_text):
        try:
            tell_tag = soup.find('p', class_='tell')
            if tell_tag and tell_tag.text.strip() != '***-****-****':
                return tell_tag.text.replace(' ', '').replace('-', ''), True
        except:
            pass
        return find_phone_num(description_text), False

    def _check_fraud(self, phone_num):
        if not phone_num:
            return False
        try:
            response = requests.post(self.api_url, json={"keyword_type": "phone", "keyword": phone_num, "add_info": ""}, headers={'X-TheCheat-ApiKey': self.api_key})
            data = response.json()
            response_temp = thecheatapi.decrypt(data['content'], self.enc_key)
            return json.loads(response_temp)['caution'] == 'Y' if response_temp else False
        except Exception as e:
            print("API request Error:", e)
            return False
    
    def _extract_product_info(self, soup):
        product_detail = soup.find('div', class_="product_detail")
        if not product_detail:
            return {}
        profile = soup.find('div', class_="profile_area")
        detail_box = product_detail.find('div', class_="product_detail_box")
        tags_to_find = ['상품 상태', '결제 방법', '배송 방법', '거래 지역']
        
        all_dl = soup.find_all('dl', class_='detail_list')
        results = {tag: (next((dl.find('dd').get_text(strip=True) for dl in all_dl if dl.find('dt') and dl.find('dt').get_text(strip=True) == tag), None)) for tag in tags_to_find}
        
        return {
            "product_name": detail_box.find('p', class_='ProductName').text,
            "product_price": detail_box.find('div', class_="ProductPrice").text,
            "membership": profile.find('em', class_='nick_level').text,
            "post_date": profile.find('div', class_="article_info").find('span', class_='date').text,
            "product_state": results['상품 상태'],
            "trade": results['결제 방법'],
            "delivery": results['배송 방법'],
            "region": results['거래 지역']
        }
    
    def _save_to_db(self, product_info):
        conn = connectDB()
        product_id = insert_product(conn, "cellphone", **product_info)
        close_connection(conn)
        product_info["product_id"] = product_id
    
    def _save_images(self, soup, product_id):
        for idx, img in enumerate(soup.find_all('img', class_="se-image-resource"), start=1):
            try:
                image_bytes = imageToS3.download_image(img['src'])
                imageToS3.upload_to_s3(self.bucket_name, image_bytes, f'cellphone/{product_id}_{idx}.jpg')
            except Exception as e:
                print("Image Upload Error:", e)

if __name__ == "__main__":
    cellphone_urls = [
        "https://cafe.naver.com/ArticleList.nhn?search.clubid=10050146&search.menuid=339&search.boardtype=L",
        "https://cafe.naver.com/ArticleList.nhn?search.clubid=10050146&search.menuid=427&search.boardtype=L",
        "https://cafe.naver.com/ArticleList.nhn?search.clubid=10050146&search.menuid=749&search.boardtype=L",
        "https://cafe.naver.com/ArticleList.nhn?search.clubid=10050146&search.menuid=424&search.boardtype=L"
    ]
    bucket_name = "c2c-trade-image"
    url_cache = URLCache(200)

    while True:
        cellphone = Cellphone(cellphone_urls, bucket_name)
        new_posts = [cellphone.jungo_url + url for url in utils.listUp(cellphone_urls) if not url_cache.is_cached(url)]
        for post_url in new_posts:
            cellphone._dynamic_crawl(post_url)
            if len(new_posts) >= 1000:
                break
        cellphone.driver.quit()
        time.sleep(randint(30, 60))
