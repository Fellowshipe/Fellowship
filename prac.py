import pandas as pd
import re
from dbControl.connect_db import connectDB

# 전화번호 패턴을 정규 표현식으로 정의 (공백 포함)
phone_pattern = re.compile(
    r'(\+?82[-.\s]?)?([01영일이삼사오육칠팔구O공０-９]{3})[-.\s]*([0-9영일이삼사오육칠팔구O공０-９]{3,4})[-.\s]*([0-9영일이삼사오육칠팔구O공０-９]{4})'
)

# 한글 및 영어 대체 숫자 사전
digit_replacements = {
    '영': '0', '일': '1', '이': '2', '삼': '3', '사': '4',
    '오': '5', '육': '6', '칠': '7', '팔': '8', '구': '9',
    'O': '0', 'I': '1', 'o': '0', 'l': '1', '공': '0',
    '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
    '５': '5', '６': '6', '７': '7', '８': '8', '９': '9'
}

def replace_korean_digits(text):
    for korean_digit, num in digit_replacements.items():
        text = text.replace(korean_digit, num)
    return text

def find_phone_number(text):
    # 한글 및 전각 숫자를 아라비아 숫자로 변환
    normalized_text = replace_korean_digits(text)
    
    # 정규 표현식을 사용하여 전화번호 찾기
    match = phone_pattern.search(normalized_text)
    
    if match:
        number = ''.join(match.groups()[1:]).replace('-', '').replace(' ', '').replace('.', '')
        if number.startswith('010') and len(number) == 11:
            return number
    return None

# 데이터베이스에서 데이터를 가져오는 쿼리
# query = "SELECT id, description FROM tickets ORDER BY id ASC LIMIT 50"
# df = pd.read_sql_query(query, connectDB())

# # 전화번호 컬럼 추가
# df['phone_number'] = df['description'].apply(find_phone_number)

# print(df)