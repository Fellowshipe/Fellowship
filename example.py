import requests
import thecheatapi
import json
import os

from dotenv import load_dotenv

load_dotenv()

api_url = os.getenv('THE_CHEAT_URL')
api_key = os.getenv('X-TheCheat-ApiKey')
enc_key = os.getenv('ENC_KEY')

request_data = {
                    "keyword_type": "phone",
                    "keyword":'01044440000',
                    "add_info": ""
                }

headers = {
    'X-TheCheat-ApiKey': api_key
}

# 더치트 API 요청 보내기
try:
    response = requests.post(api_url, json=request_data, headers=headers)
    data = response.json()
    response_temp = thecheatapi.decrypt(data['content'], enc_key)
except Exception as e:
    print("API request Error:", {e})

print(response_temp)

#사기 피해 여부
fraud_check = json.loads(response_temp)['caution']
if fraud_check == 'Y':
    fraud_check = True 
else:
    fraud_check = False

print(fraud_check)