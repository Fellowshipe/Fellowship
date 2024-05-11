import requests
import time

def request(url):
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