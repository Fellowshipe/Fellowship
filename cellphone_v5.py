import os
import csv
import gc
import uuid
import time
import json
import tempfile
import requests
from random import randint
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from dotenv import load_dotenv

from JungoNara import JungoNara
from utils.URLCache import URLCache
import utils.utils as utils
import utils.imageToS3 as imageToS3  # 다운로드 함수만 사용
import utils.thecheatapi as thecheatapi
from src.data_processing import find_phone_num

# 로컬 디렉토리 생성
os.makedirs("./data", exist_ok=True)
os.makedirs("./images", exist_ok=True)

CSV_PATH = "./data/cellphone_products.csv"

# CSV 헤더 작성 (처음 실행 시에만)
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "product_id", "product_name", "product_price", "membership", "post_date",
            "product_status", "trade", "delivery", "region",
            "description", "phone_num", "is_fraud", "is_find"
        ])

class Cellphone(JungoNara):
    def __init__(self, base_url, delay_time=None, saving_html=False):
        super().__init__(delay_time, saving_html)
        self.base_url = base_url
        self.jungo_url = "https://cafe.naver.com"

        load_dotenv()
        self.api_url = os.getenv('THE_CHEAT_URL')
        self.api_key = os.getenv('X-TheCheat-ApiKey')
        self.enc_key = os.getenv('ENC_KEY')

        user_data_dir = tempfile.mkdtemp()
        print("[DEBUG] Using user-data-dir:", user_data_dir)
        options = webdriver.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--headless")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f"--user-data-dir={user_data_dir}")
        service = Service(executable_path="/usr/local/bin/chromedriver")

        self.driver = webdriver.Chrome(service=service, options=options)

    def _dynamic_crawl(self, url: str) -> str:
        print(f"[INFO] 수집 시도 중: {url}")
        if not url.startswith(self.jungo_url):
            return

        self.driver.get(url)
        time.sleep(3)

        if isinstance(self.delay_time, float): time.sleep(self.delay_time)
        elif isinstance(self.delay_time, tuple): time.sleep(float(randint(*self.delay_time)))

        try:
            wait = WebDriverWait(self.driver, 10)
            iframe = wait.until(EC.presence_of_element_located((By.ID, "cafe_main")))
            self.driver.switch_to.frame(iframe)
        except:
            print("[WARNING] iframe 로딩 실패")
            return

        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        se_module = soup.find_all('div', class_="se-section se-section-text se-l-default")

        span_texts = [span.get_text(strip=True)
                      for module in se_module
                      for span in module.select('div > p > span')]
        description_text = "\n".join(span_texts)

        is_tell = False
        try:
            tell_tag = soup.find('p', class_='tell')
            if tell_tag.text.strip() == '***-****-****':
                find_phone = find_phone_num(description_text)
                if not find_phone:
                    print("[INFO] 안심번호 사용 중이며 설명에도 번호 없음")
                    return
            is_tell = True
            find_phone = find_phone_num(description_text)
        except:
            find_phone = find_phone_num(description_text)
            if not find_phone:
                print("[INFO] 번호 추출 실패")
                return

        headers = { 'X-TheCheat-ApiKey': self.api_key }
        fraud_check = 'N'
        found_fraud_check = 'N'
        cleaned_number = tell_tag.text.replace(' ', '').replace('-', '') if is_tell else None

        try:
            if cleaned_number:
                resp = requests.post(self.api_url, json={"keyword_type": "phone", "keyword": cleaned_number}, headers=headers)
                decrypted = thecheatapi.decrypt(resp.json()['content'], self.enc_key)
                fraud_check = json.loads(decrypted).get('caution', 'N')

            if find_phone:
                resp = requests.post(self.api_url, json={"keyword_type": "phone", "keyword": find_phone}, headers=headers)
                decrypted = thecheatapi.decrypt(resp.json()['content'], self.enc_key)
                found_fraud_check = json.loads(decrypted).get('caution', 'N')
        except:
            print("[ERROR] 더치트 API 요청 실패")
            return

        is_fraud = fraud_check == 'Y' or found_fraud_check == 'Y'
        product_detail = soup.find('div', class_="product_detail")
        product_detail_box = product_detail.find('div', class_="product_detail_box")
        images = soup.find_all('img', class_="se-image-resource")
        profile = soup.find('div', class_="profile_area")

        tags = ['상품 상태', '결제 방법', '배송 방법', '거래 지역']
        results = {}
        all_dl = soup.find_all('dl', class_='detail_list')
        for tag in tags:
            dl = next((dl for dl in all_dl if dl.find('dt') and dl.find('dt').get_text(strip=True) == tag), None)
            results[tag] = dl.find('dd').get_text(strip=True) if dl and dl.find('dd') else None

        product_id = str(uuid.uuid4())
        product_name = product_detail_box.find('p', class_='ProductName').text
        product_price = product_detail_box.find('div', class_="ProductPrice").text
        membership = profile.find('em', class_='nick_level').text
        post_date = profile.find('div', class_="article_info").find('span', class_='date').text
        phone_num = f"{cleaned_number} {find_phone}" if cleaned_number and find_phone else (cleaned_number or find_phone)
        is_find = cleaned_number is None and find_phone is not None

        with open(CSV_PATH, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                product_id, product_name, product_price, membership, post_date,
                results['상품 상태'], results['결제 방법'], results['배송 방법'], results['거래 지역'],
                description_text, phone_num, is_fraud, is_find
            ])
        print(f"[SUCCESS] 수집 완료: {product_name} | {product_price}")

        temp_num = 1
        for img in images:
            try:
                url = img['src']
                image_bytes = imageToS3.download_image(url)
                local_path = f"./images/{product_id}_{temp_num}.jpg"
                with open(local_path, "wb") as f:
                    f.write(image_bytes)
                temp_num += 1
            except Exception as e:
                print(f"[WARNING] 이미지 저장 실패: {e}")

        self.driver.switch_to.default_content()
        gc.collect()
