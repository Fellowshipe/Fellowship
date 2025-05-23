from random import randint

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import gc

import json
import os
from dotenv import load_dotenv
import requests
import time
from selenium import webdriver
import tempfile
import csv
import uuid
import os

from JungoNara import JungoNara
# from src.dbControl.connect_db import connectDB
# from src.dbControl.create_product_table import create_product_table
# from src.dbControl.close_connection import close_connection
# from src.dbControl.insert_product import insert_product
from selenium.webdriver.chrome.service import Service

from utils.URLCache import URLCache
import utils.utils as utils
import utils.imageToS3 as imageToS3
import utils.thecheatapi as thecheatapi
from src.data_processing import find_phone_num

os.makedirs("./data", exist_ok=True)
os.makedirs("./images", exist_ok=True)

product_id = str(uuid.uuid4())  # 고유 ID


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
        
        user_data_dir = tempfile.mkdtemp()
        print("[DEBUG] Using user-data-dir:", user_data_dir)
        options = webdriver.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument('--headless')
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f"--user-data-dir={tempfile.mkdtemp()}")
        service = Service(executable_path="/usr/local/bin/chromedriver")  # 명시적으로 지정

        self.driver = webdriver.Chrome(options=options)

    def _dynamic_crawl(self, url: str) -> str:        
        assert url.startswith(self.jungo_url), "Given url does not seem to be from cellphoe category."

        self.driver.get(url)
       
        time.sleep(3)

        if isinstance(self.delay_time, float): time.sleep(self.delay_time)
        elif isinstance(self.delay_time, tuple): time.sleep(float(randint(self.delay_time[0], self.delay_time[1])))
        elif self.delay_time == None: pass
        else: raise TypeError("you must give delay_time float or tuple type.")
        
        # iframe이 로드될 때까지 대기
        try:
            wait = WebDriverWait(self.driver, 10)
        except Exception as e:
            print(e)

        try:
            iframe = wait.until(EC.presence_of_element_located((By.ID, "cafe_main")))
        except:
            #print("삭제된 게시물")
            return
        
        # 브라우저 변환
        try:
            self.driver.switch_to.frame(iframe)
        except Exception as e:
            print(e)

        # 페이지 소스 가져오기
        html_content = self.driver.page_source
        #print(html_content)

        soup = BeautifulSoup(html_content, 'html.parser')


        se_module = soup.find_all('div', class_="se-section se-section-text se-l-default")

        # # 상품 소개
        # 각 div 태그 내에서 'div > p > span'에 해당하는 모든 span 태그를 찾기
        span_texts = []
        for module in se_module:
            spans = module.select('div > p > span')
            for span in spans:
                span_texts.append(span.get_text(strip=True))

        # span 텍스트들을 줄바꿈 문자로 연결
        description_text = "\n".join(span_texts)


        is_tell = False

        # 없는 전화번호, 안심번호 필터링
        try:
            tell_tag = soup.find('p', class_='tell')
            #print(tell_tag.text)
            if tell_tag.text == ' ***-****-**** ':
                #print("안심번호 사용중")
                # 상품 설명에서 전화번호 찾기 
                find_phone = find_phone_num(description_text)
                if find_phone is None:
                    print("상품 설명도 존재 x")
                    return
            print("프로필에서 번호 추출 완료")
            is_tell = True
            find_phone = find_phone_num(description_text)
            if find_phone is None:
                print("프로필에서는 존재하지만 상품 설명에서는 번호 존재 X")
        except:
            print("전화번호 추출 불가")
            find_phone = find_phone_num(description_text)
            if find_phone is None:
                print("상품 설명도 존재 x")
                return
            
        # API 요청 header 설정
        headers = {
            'X-TheCheat-ApiKey': self.api_key
        }

        # 초기화
        fraud_check = 'N'
        found_fraud_check = 'N'

        # 전처리 전화번호 초기화
        cleaned_number = None

        # 전화번호 전처리
        if is_tell == True:
            cleaned_number = tell_tag.text.replace(' ', '').replace('-', '')

        # 더치트 API 요청 보내기 (프로필 전화번호)
        if cleaned_number is not None:
            # the Cheat API 요청 데이터
            request_data = {
                        "keyword_type": "phone",
                        "keyword": cleaned_number,
                        "add_info": ""
                    }
            try:
                response = requests.post(self.api_url, json=request_data, headers=headers)
                data = response.json()
                response_temp = thecheatapi.decrypt(data['content'], self.enc_key)

                # 사기 피해 여부
                if response_temp is not None:
                    fraud_check = json.loads(response_temp)['caution']
                    print("프로필", fraud_check)
            except Exception as e:
                print("API request Error:", {e})
                return
        

        # 상품 설명 전화번호 API 요청
        if find_phone is not None:
            found_request_data = {
                    "keyword_type": "phone",
                    "keyword": find_phone,
                    "add_info": ""
                 }

            # 더치트 API 요청 보내기 (찾은 전화번호)
            try:
                response = requests.post(self.api_url, json=found_request_data, headers=headers)
                data = response.json()
                found_response_temp = thecheatapi.decrypt(data['content'], self.enc_key)

                # 사기 피해 여부
                if found_response_temp is not None:
                    found_fraud_check = json.loads(found_response_temp)['caution']
                    #print("상품설명", found_fraud_check)
            except Exception as e:
                print("API request Error:", {e})
                return
        
       
        # 사기 여부 탐지
        if fraud_check == 'Y' or found_fraud_check == 'Y':
            is_fraud = True
        else:
            is_fraud = False
        

        # 상품 정보 찾기
        product_detail = soup.find('div', class_="product_detail")
        images = soup.find_all('img', class_="se-image-resource")
        profile = soup.find('div', class_="profile_area")

        product_detail_box = product_detail.find('div', class_="product_detail_box")


        # HTML 파싱을 위한 BeautifulSoup 객체 생성    # 정보를 추출할 태그 목록
        tags_to_find = ['상품 상태', '결제 방법', '배송 방법', '거래 지역']

        # 결과를 저장할 딕셔너리
        results = {}

        # 'detail_list' 클래스를 가진 모든 'dl' 태그를 찾음
        all_dl = soup.find_all('dl', class_='detail_list')


        # 각 태그 목록에 대해 처리
        for tag in tags_to_find:
            # 해당 태그가 포함된 'dl' 태그 찾기
            dl = next((dl for dl in all_dl if dl.find('dt') and dl.find('dt').get_text(strip=True) == tag), None)
            
            # 해당 'dl' 태그가 있고, 그 안에 'dd' 태그도 있으면 텍스트를 가져옴
            if dl and dl.find('dd'):
                results[tag] = dl.find('dd').get_text(strip=True)
            else:
                # 해당 태그가 없으면 결과에 None 저장
                results[tag] = None


        product_name = product_detail_box.find('p', class_='ProductName').text
        product_price = product_detail_box.find('div', class_="ProductPrice").text
        membership = profile.find('em', class_='nick_level').text
        post_date = profile.find('div', class_="article_info").find('span', class_='date').text
        product_state = results['상품 상태']
        trade = results['결제 방법']
        delivery = results['배송 방법']
        region = results['거래 지역']
        
        # 둘 다 존재하면, 띄어쓰기해서 저장
        if cleaned_number is not None and find_phone is not None:
            phone_num = cleaned_number + ' ' + find_phone
        else:
            phone_num = cleaned_number or find_phone

        # 상품에서만 존재하는지 아닌지 확인
        if cleaned_number is None and find_phone is not None:
            is_find = True
        else:
            is_find = False

        #print("db 저장 전화번호 출력", phone_num)
        #print("is_find", is_find)

        # # 데이터베이스 연결
        # conn = connectDB()

        # # 상품 데이터 삽입
        # # fraud check -> 최근 3개월
        # # MFCC, RNN/LSTM를 활용한 연구 방법을 사용
        # try:
        #     product_id = insert_product(conn, 
        #                                 "cellphone", 
        #                                 product_name, 
        #                                 product_price, 
        #                                 membership,
        #                                 post_date, 
        #                                 product_state, 
        #                                 trade, 
        #                                 delivery, 
        #                                 region, 
        #                                 description_text, 
        #                                 phone_num, 
        #                                 is_fraud,
        #                                 is_find
        #                             )
        # finally:
        #     close_connection(conn)
        product_id = str(uuid.uuid4())  # 고유 ID

        # CSV 저장
        with open("./data/cellphone_products.csv", "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                product_id,
                product_name, 
                product_price, 
                membership,
                post_date, 
                product_state, 
                trade, 
                delivery, 
                region, 
                description_text, 
                phone_num, 
                is_fraud,
                is_find
            ])
            
        temp_num = 1
        # 이미지 크롤링
        for img in images:
            try:
                url = img['src']
                image_bytes = imageToS3.download_image(url)
                file_name = f'cellphone/{product_id}_{temp_num}.jpg'
                temp_num += 1
                local_path = f"./images/{product_id}_{temp_num}.jpg"
                with open(local_path, "wb") as f:
                    f.write(image_bytes)

                # 잔여 메모리 처리
                del image_bytes
                gc.collect()
            except requests.RequestException as e:
                print(f"Failed to download {url}: {e}")
            except Exception as e:
                print(f"Failed to upload to S3: {e}")

        #브라우저 초기화
        self.driver.switch_to.default_content()

        # 메모리 정리
        del soup
        del html_content
        del images
        del description_text
        gc.collect()

if __name__ == "__main__":
    cellphone_url = [
        "https://cafe.naver.com/ArticleList.nhn?search.clubid=10050146&search.menuid=339&search.boardtype=L",
        "https://cafe.naver.com/ArticleList.nhn?search.clubid=10050146&search.menuid=427&search.boardtype=L",
        "https://cafe.naver.com/ArticleList.nhn?search.clubid=10050146&search.menuid=749&search.boardtype=L",
        "https://cafe.naver.com/ArticleList.nhn?search.clubid=10050146&search.menuid=424&search.boardtype=L"
    ]
    bucket_name = "c2c-trade-image"
    url_cache = URLCache(200)

    # ✅ DB 연결 및 테이블 생성 (최초 1회)
    conn = connectDB()
    create_product_table(conn)
    conn.close()

    crawl_count = 0  # 게시물 크롤링 카운트 초기화

    try:
        while True:
            # Cellphone 객체 생성
            cellphone = Cellphone(cellphone_url, bucket_name)

            try:
                new_posts = []

                # 주어진 URL 목록을 순회하면서 캐시에 없는 URL만 처리
                for url in utils.listUp(cellphone_url):
                    full_url = cellphone.jungo_url + url
                    if not url_cache.is_cached(url):
                        new_posts.append(full_url)  # 캐시에 없는 URL에 접두어를 붙여 new_posts에 추가
                        url_cache.add_to_cache(url)  # 캐시에 URL을 추가

                for post_url in new_posts:
                    cellphone.dynamic_crawl(post_url)
                    crawl_count += 1  # 게시물 크롤링 횟수 증가

                    # 1000개 크롤링 후 객체 재생성
                    if crawl_count >= 1000:
                        print("1000개 게시물 크롤링 완료. 객체를 재생성합니다.")
                        cellphone.driver.quit()  # Webdriver 종료
                        crawl_count = 0  # 크롤링 카운트 초기화
                        break  # 객체 재생성을 위해 루프 탈출

            finally:
                cellphone.driver.quit()  # Webdriver 종료
                del cellphone
                gc.collect()
                cellphone = Cellphone(cellphone_url, bucket_name)

            time.sleep(randint(30, 60))  # 1분마다 새 게시물 확인

    except KeyboardInterrupt:
        print("크롤링 중단됨.")
