from random import randint

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

import json
import os
from dotenv import load_dotenv
import requests
import time
from selenium import webdriver

from JungoNara import JungoNara
from dbControl.connect_db import connectDB
from dbControl.close_connection import close_connection
from dbControl.insert_product import insert_product

from URLCache import URLCache
import utils
import imageToS3
import thecheatapi

class Cellphone(JungoNara):
    def __init__(self, base_url, bucket_name, delay_time=None, saving_html=False):
        super().__init__(delay_time, saving_html)
        self.base_url = base_url
        self.jungo_url = "https://cafe.naver.com"
        self.bucket_name = bucket_name

        self.api_url = os.getenv('THE_CHEAT_URL')
        self.api_key = os.getenv('X-TheCheat-ApiKey')
        self.enc_key = os.getenv('ENC_KEY')

        options = webdriver.ChromeOptions()
        #options.add_argument('Chrome/123.0.6312.122')
        options.add_argument("--no-sandbox")
        options.add_argument('--headless')
        self.driver = webdriver.Chrome(options=options)

    def _dynamic_crawl(self, url: str) -> str:        
        assert url.startswith(self.jungo_url), "Given url does not seem to be from cellphone category."

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
            print("삭제된 게시물")
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

        # 없는 전화번호, 안심번호 필터링
        try:
            tell_tag = soup.find('p', class_='tell')
            if tell_tag.text == ' ***-****-**** ':
                print("안심번호 사용중")
                return
                
            else:
                print(tell_tag.text)
        except:
            print("전화번호 추출 불가")
            return

        # API 변수 읽기
        load_dotenv()

        cleaned_number = tell_tag.text.replace(' ', '').replace('-', '')

        request_data = {
                    "keyword_type": "phone",
                    "keyword": cleaned_number,
                    "add_info": ""
                }

        headers = {
            'X-TheCheat-ApiKey': self.api_key
        }

        # 더치트 API 요청 보내기
        try:
            response = requests.post(self.api_url, json=request_data, headers=headers)
            data = response.json()
            response_temp = thecheatapi.decrypt(data['content'], self.enc_key)
        except Exception as e:
            print("API request Error:", {e})

        # 사기 피해 여부
        fraud_check = json.loads(response_temp)['caution']
        if fraud_check == 'Y':
            fraud_check = True
        else:
            fraud_check = False

        # 상품 정보 찾기
        product_detail = soup.find('div', class_="product_detail")
        se_module = soup.find_all('div', class_="se-section se-section-text se-l-default")
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
            
    
        # # 상품 소개
        # 각 div 태그 내에서 'div > p > span'에 해당하는 모든 span 태그를 찾기
        span_texts = []
        for module in se_module:
            spans = module.select('div > p > span')
            for span in spans:
                span_texts.append(span.get_text(strip=True))

        # span 텍스트들을 줄바꿈 문자로 연결
        description_text = "\n".join(span_texts)

        # 데이터베이스 연결
        conn = connectDB()

        # 상품 데이터 삽입
        product_id = insert_product(conn, product_name, product_price, membership,
                       post_date, product_state, trade, delivery, region, description_text
                       )
        
        # 연결 종료
        close_connection(conn)
        
        # 
        temp_num = 1
        # 이미지 크롤링
        for img in images:
            try:
                url = img['src']
                image_bytes = imageToS3.download_image(url)
                file_name = f'imgs/{product_id}_{temp_num}.jpg'
                temp_num += 1
                imageToS3.upload_to_s3(self.bucket_name, image_bytes, file_name)
            except requests.RequestException as e:
                print(f"Failed to download {url}: {e}")
            except Exception as e:
                print(f"Failed to upload to S3: {e}")
        # 브라우저 초기화
        self.driver.switch_to.default_content()
    
if __name__ == "__main__":
    #driver = utils.get_driver() # WebDriver 초기화
    cellphone_url = "https://cafe.naver.com/ArticleList.nhn?search.clubid=10050146&search.menuid=1156&search.boardtype=L"
    bucket_name = "c2c-trade-image"

    Cellphone = Cellphone(cellphone_url, bucket_name)
    url_cache = URLCache()

    #Cellphone.dynamic_crawl(driver, 'https://cafe.naver.com/ArticleRead.nhn?clubid=10050146&page=1&menuid=1156&boardtype=L&articleid=1056735750&referrerAllArticles=false')

    try:
        while True:
            # new_posts 초기화
            new_posts = []

            # 주어진 URL 목록을 순회하면서 캐시에 없는 URL만 처리
            for url in utils.listUp(cellphone_url):
                full_url = Cellphone.jungo_url + url
                if not url_cache.is_cached(url):
                    new_posts.append(full_url)  # 캐시에 없는 URL에 접두어를 붙여 new_posts에 추가
                    url_cache.add_to_cache(url)  # 캐시에 URL을 추가

            for post_url in new_posts:
                print(f"Crawling {post_url}")
                Cellphone.dynamic_crawl(post_url)
            time.sleep(3) # 1분마다 새 게시물 확인
    finally:
        Cellphone.driver.quit() # Webdriver 종료