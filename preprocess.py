import pandas as pd
import string
from dbControl.connect_db import connectDB
from db_to_df import fetch_data_from_db
import os

def process_all_dbms_data(df):
    # 데이터프레임 복사
    df_processed = df.copy()
    
    # 문자열에서 "원"과 콤마(,)를 제거하고 정수형으로 변환
    df_processed['price'] = df_processed['price'].str.replace('원', '')  # "원" 제거
    df_processed['price'] = df_processed['price'].str.replace(',', '')   # 콤마 제거
    
    # 각 가격에서 0의 비율을 계산하는 함수
    def calculate_zero_ratio(price):
        price_str = str(price)
        zero_count = price_str.count('0')
        total_count = len(price_str)
        return zero_count / total_count if total_count > 0 else 0
    
    # 새로운 컬럼 추가
    df_processed['zero_price_ratio'] = df_processed['price'].apply(calculate_zero_ratio)
    
    # int 변환
    df_processed['price'] = df_processed['price'].astype(int)
    
    # 원핫 인코딩을 할 컬럼 리스트
    columns_to_encode = ['member_level', 'product_status', 'payment_method', 'shipping_method']
    
    # 원핫 인코딩 수행
    df_processed = pd.get_dummies(df_processed, columns=columns_to_encode)
    
    # transaction_region -> None은 0, 데이터 있으면 1
    df_processed['transaction_region'] = df_processed['transaction_region'].apply(lambda x: 0 if pd.isnull(x) else 1)
    
    # post_date -> 요일 변수 생성, 시간대 변수 생성
    df_processed['post_date'] = pd.to_datetime(df_processed['post_date'])
    df_processed['post_day_of_week'] = df_processed['post_date'].dt.dayofweek  # 요일 (0=Monday, 1=Tuesday, ..., 6=Sunday)
    df_processed['post_hour'] = df_processed['post_date'].dt.hour  # 시간대 (0 ~ 23)

    # description -> 글자수 변수 생성
    df_processed['description_length'] = df_processed['description'].apply(len)

    # description -> 특수문자 및 숫자의 비율
    def calculate_ratios(text):
        total_chars = len(text)
        if total_chars == 0:
            return 0, 0
        special_chars = sum(1 for char in text if char in string.punctuation)
        digits = sum(1 for char in text if char.isdigit())
        special_ratio = special_chars / total_chars
        digit_ratio = digits / total_chars
        return special_ratio, digit_ratio
    
    df_processed['description_special_ratio'], df_processed['description_digit_ratio'] = zip(*df_processed['description'].apply(calculate_ratios))

    return df_processed



if __name__ == "__main__":
    query = "SELECT * FROM cellphone"
    df = pd.read_sql_query(query, connectDB())
    print(df)
    
    df_processed = process_all_dbms_data(df)
    print(df_processed)