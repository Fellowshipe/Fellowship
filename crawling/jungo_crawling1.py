from jungo_crawling2 import listUp
import pandas as pd


URL = "https://cafe.naver.com/ArticleList.nhn?search.clubid=10050146&search.menuid=339&search.boardtype=L&search.totalCount=151&search.cafeId=10050146&search.page="


data = [] 
page = 10

# 원하는 페이지까지 크롤링을 수행
for i in range(1, page + 1):
    print(i)

    temp_url = URL + str(i)

    # 셀러 유저가 아닌 유저들의 게시물 url정보를 크롤링
    map = listUp(temp_url)

    data.extend(map)
    

df = pd.DataFrame(data)

df.to_csv('url2.csv', index=False)
