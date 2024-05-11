import pandas as pd
from random import randint

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

from urllib.request import urlopen

import time

from JungoNara import JungoNara
import utils

class Cellphone(JungoNara):
    def __init__(self, base_url, delay_time=None, saving_html=False):
        super().__init__(delay_time, saving_html)
        self.base_url = base_url
        self.jungo_url = "https://cafe.naver.com"

    def _dynamic_crawl(self, driver, url: str) -> str:
        assert url.startswith(self.jungo_url), "Given url does not seem to be from cellphone category."
        
        driver.get(url)

        # sleep
        if isinstance(self.delay_time, float): time.sleep(self.delay_time)
        elif isinstance(self.delay_time, tuple): time.sleep(float(randint(self.delay_time[0], self.delay_time[1])))
        elif self.delay_time == None: pass
        else: raise TypeError("you must give delay_time float or tuple type.")
        
        # iframe이 로드될 때까지 대기
        # iframe이 로드될 때까지 대기
        wait = WebDriverWait(driver, 10)
        try:
            iframe = wait.until(EC.presence_of_element_located((By.ID, "cafe_main")))
        except:
            print("삭제된 게시물")
            return
        # 브라우저 변환
        driver.switch_to.frame(iframe)

        # 페이지 소스 가져오기
        html_content = driver.page_source

        # BeatifulSoup 객체 생성
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

        product_detail = soup.find('div', class_="product_detail")
        #se_module = soup.find_all('div', class_="se-section se-section-text se-l-default")
        images = soup.find_all('img', class_="se-image-resource")
        profile = soup.find('div', class_="profile_area")

        product_detail_box = product_detail.find('div', class_="product_detail_box")
        commercialDetail = product_detail.select('div > div.section > dl')


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

        # for i, v in results.items():
        #     print(i, v)

        map = pd.DataFrame([{
                "product_name": product_detail_box.find('p', class_='ProductName').text,
                "product_price": product_detail_box.find('div', class_="ProductPrice").text,
                "membership": profile.find('em', class_='nick_level').text,
                "post_data": profile.find('div', class_="article_info").find('span', class_='date').text,
                "product_state": results['상품 상태'],
                "trade": results['결제 방법'],
                "delivery" : results['배송 방법'],
                "region" : results['거래 지역']
            }])
        for i, v in map.items():
            print(i, v)
        # 데이터 추가
        #final_df = pd.concat([final_df, map], ignore_index=True)


        # # 상품 소개
        # product_dec = se_module.select('div > p > span')
        # for p in product_dec:
        #     print(p.text)

        # 이미지 temp
        t = 0

        # 이미지 크롤링
        # for img in images:
        #     try:
        #         url = img['src']
        #         try:
        #             with urlopen(url) as f:
        #                 #print(url)
        #                 with open('./images/img' + str(i) + '_' + str(t) + '.jpg', 'wb') as h:
        #                     img = f.read()
        #                     h.write(img)
        #                     t += 1
        #         except:
        #             continue
        #     except:
        #         continue

        # 브라우저 초기화
        driver.switch_to.default_content()
    
if __name__ == "__main__":
    driver = utils.get_driver() # WebDriver 초기화
    cellphone_url = "https://cafe.naver.com/ArticleList.nhn?search.clubid=10050146&search.menuid=1156&search.boardtype=L"
    Cellphone = Cellphone(cellphone_url)

    try:
        while True:
            new_posts = utils.listUp(cellphone_url)
            new_posts = [Cellphone.jungo_url + url for url in new_posts if not utils.check_visited(url)]

            for post_url in new_posts:
                print(f"Crawling {post_url}")
                Cellphone.dynamic_crawl(driver, post_url)
                utils.check_visited(post_url)
            
            time.sleep(60) # 1분마다 새 게시물 확인
    finally:
        driver.quit() # Webdriver 종료