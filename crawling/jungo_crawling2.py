from bs4 import BeautifulSoup

from jungo_request import request

SELLER_MEM = "https://cafe.pstatic.net/levelicon/1/1_150.gif"


def listUp(URL):
    temp_arr = []

    html = request(URL)

    soup = BeautifulSoup(html, 'html.parser')

    tr_arr = soup.select('#main-area > div.article-board:not([id="upperArticleList"]) > table > tbody > tr')

    for tr in tr_arr:
        a_tag = tr.select_one('td.td_article a')
        # tr 태그 내부에 CSS Selector 로 접근해서 'a' 태그를 가져옴 ('a' 는 anchor 의 줄임)
        # a 태그에는 우리가 원하는 '제목' 이 들어있음

        name_tag = tr.select_one('td.td_name > div > table > tr > td > span img')
        
        try: 
            if name_tag["src"] == SELLER_MEM:
                continue
        except:
            # 확인 불가
            print("확인 불가")
            continue
        
        map = {
            "title": a_tag.text.strip(),
            "url": a_tag["href"],
            "is_checked": False,
            "is_new_item": True
        }

        # 나의 데이터목록을 for loop 으로 확인
        for element in temp_arr:
            # 기존에 확인했던 element 와 방금 새로 확인한 map 데이터가 같은지 비교
            if element["url"] == map["url"]:
                map["is_new_item"] = False
                # url 값이 같다면 새로운 데이터가 아님!
                break

        # 만약 새로운 데이터라면
        if map["is_new_item"]:
            temp_arr.insert(0, map)
            # 나의 데이터목록에 추가

    return temp_arr
        
        


