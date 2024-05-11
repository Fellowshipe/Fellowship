import pandas as pd

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

# brew services stop mongodb-community@5.0
from pymongo import MongoClient
import gridfs
import base64

from urllib.request import urlopen
import time


# URL이 저장되어 있는 CSV 파일 경로
file_path = "./url2.csv"
BASE_URL = "https://cafe.naver.com"

#필요한 열 지정
cols_to_use = ['url']

#read_csv 함수를 사용해서 특정 열만 불러오기
df = pd.read_csv(file_path, usecols=cols_to_use)

# WebDriver 경로 설정
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--no-sandbox")

driver = webdriver.Chrome(options=options)

# 상품에 대한 정보 
final_df = pd.DataFrame(columns=['product_name', 'product_price', 'product_state', 'trade', 'delivery', 'region'])

# MongoDB 클라이언트 설정
client = MongoClient('mongodb://localhost:27017/')
db = client['fellowship_tutorial']
fs = gridfs.GridFS(db)

#driver.get("https://cafe.naver.com/ArticleRead.nhn?clubid=10050146&page=1&menuid=339&boardtype=L&articleid=1051713563&referrerAllArticles=false")
# 로딩 대기

for i in range(len(df)):
    driver.get(BASE_URL + df.iloc[i, 0])
    # print(BASE_URL + df.iloc[i,0])

    time.sleep(3)



    # iframe이 로드될 때까지 대기
    wait = WebDriverWait(driver, 10)
    try:
        iframe = wait.until(EC.presence_of_element_located((By.ID, "cafe_main")))
    except:
        print("삭제된 게시물")
        continue

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
            continue
        else:
            print(tell_tag.text)
    except:
        print("전화번호 추출 불가")
        continue

    product_detail = soup.find('div', class_="product_detail")
    #se_module = soup.find_all('div', class_="se-section se-section-text se-l-default")
    images = soup.find_all('img', class_="se-image-resource")

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
            "product_state": results['상품 상태'],
            "trade": results['결제 방법'],
            "delivery" : results['배송 방법'],
            "region" : results['거래 지역']
        }])
    

    # 데이터 추가
    final_df = pd.concat([final_df, map], ignore_index=True)


    # # 상품 소개
    # product_dec = se_module.select('div > p > span')
    # for p in product_dec:
    #     print(p.text)

    # 이미지 temp
    t = 0

    # 이미지 크롤링
    for img in images:
        try:
            url = img['src']
            try:
                with urlopen(url) as f:
                    #print(url)
                    with open('./images/img' + str(i) + '_' + str(t) + '.jpg', 'wb') as h:
                        img = f.read()
                        h.write(img)
                        t += 1
            except:
                continue
        except:
            continue

    # 브라우저 초기화
    driver.switch_to.default_content()

# 브라우저 종료
driver.quit()

final_df.to_csv('productDetail_example.csv', index=False)
