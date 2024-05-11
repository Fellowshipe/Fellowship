from functools import lru_cache

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from bs4 import BeautifulSoup
import requests
import time


@lru_cache(maxsize=50)
def check_visited(url):
    """
        이 함수는 주어진 URL이 이미 방문했는지 체크함.
        LRU 캐시에 저장되어 있으면 True, 아니면 False를 반환하고 저장함.

        maxsize = 50이므로 최대 50개의 게시물의 url을 캐시에 저장할 수 있음.
    """
    return False



def get_driver():
    """WebDriver 인스턴스를 생성하고 반환함."""
    options = webdriver.ChromeOptions()
    #options.add_argument('Chrome/123.0.6312.122')
    options.add_argument('log-level=3')
    options.add_argument('headless')
    driver = webdriver.Chrome(options=options)

    return driver


def listUp(URL):
    temp_arr = []
    SELLER_MEM = "https://cafe.pstatic.net/levelicon/1/1_150.gif"

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
        


