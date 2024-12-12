from selenium import webdriver

from bs4 import BeautifulSoup
import requests
import time

from src.data_processing.find_phone_num import find_phone_number

def get_driver():
    """WebDriver 인스턴스를 생성하고 반환함."""
    options = webdriver.ChromeOptions()
    #options.add_argument('Chrome/123.0.6312.122')
    options.add_argument("--no-sandbox")
    #options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)

    return driver


def listUp(URL_LIST):
    temp_arr = []
    SELLER_MEM = "https://cafe.pstatic.net/levelicon/1/1_150.gif"

    for URL in URL_LIST:
        try:
            html = request(URL)
        except Exception as e:
            print("request error:", e)
            return []  # 또는 적절한 에러 처리


        soup = BeautifulSoup(html, 'html.parser')

        tr_arr = soup.select('#main-area > div.article-board:not([id="upperArticleList"]) > table > tbody > tr')

        for tr in tr_arr:
            a_tag = tr.select_one('td.td_article a')
            # tr 태그 내부에 CSS Selector 로 접근해서 'a' 태그를 가져옴 ('a' 는 anchor 의 줄임)
            # a 태그에는 우리가 원하는 '제목' 이 들어있음

            name_tag = tr.select_one('td.td_name > div > table > tr > td > span img')
            
            try: 
                if name_tag["src"] == SELLER_MEM:
                    print(a_tag["href"], "셀러회원")
                    continue
            except:
                # 확인 불가
                print("확인 불가")
                continue
            
            url = a_tag["href"]
            print(url)
            temp_arr.insert(0, url)

    return temp_arr
    

def request(url):
    response = None
    try:
        response = requests.request('GET', url)
    except Exception as e:
        print("오류 발생")
        print(e)
        print("5초 후 재시도")  
        time.sleep(5)
        request(url)
    
    if response.status_code != 200:
        print(response.status_code)
        raise f'${response.status_code}'

    return response.text
        

def find_phone_num_in_description(description_text):
    return find_phone_number(description_text)