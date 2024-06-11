# 데이터 전처리

## 정형 데이터

### 수치형 변수 -> Standardization
### 1) 가격 : price
### 2) 가격에서 0의 비율 : zero_price_ratio
### 3) 본문 글자수 : description_length
### 4) 본문 특수문자, 숫자 비율 : description_special_ratio, description_digit_ratio
### 5) 이미지 개수 : image_count

### 범주형 변수 -> One Hot Encoding or Assign
### 6) 회원 등급 : member_level
### 7) 작성 요일/시간대 : post_day_of_week, post_hour
### 8) 상품 상태 : product_status
### 9) 결제 방법 : payment_method
### 10) 배송 방법 : shipping_method
### 11) 거래 지역 공개 여부 : transaction_region
### 12) 연락처 공개 여부 : is_find

import pandas as pd
import string
import boto3
from konlpy.tag import Okt
import os
from dbControl.connect_db import connectDB

# Initialize the tokenizer
okt = Okt()

# 정형 데이터 전처리
def process_tabular_data(df):
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
    
    # transaction_region -> None은 0, 데이터 있으면 1
    df_processed['transaction_region'] = df_processed['transaction_region'].apply(lambda x: 0 if pd.isnull(x) else 1)
    
    # post_date -> 요일 변수 생성, 시간대 변수 생성
    df_processed['post_date'] = pd.to_datetime(df_processed['post_date'])
    df_processed['post_day_of_week'] = df_processed['post_date'].dt.dayofweek  # 요일 (0=Monday, 1=Tuesday, ..., 6=Sunday)
    df_processed['post_hour'] = df_processed['post_date'].dt.hour  # 시간대 (0 ~ 23)

    # 원핫 인코딩을 할 컬럼 리스트
    columns_to_encode = ['member_level', 'product_status', 'payment_method', 'post_day_of_week', 'post_hour']

    # 문자열을 리스트로 변환하는 함수
    def convert_shipping_method_to_list(shipping_method):
        if pd.isnull(shipping_method):
            return []
        return shipping_method.split(', ')
    
    # shipping_method 컬럼을 리스트 형태로 변환
    df_processed['shipping_method'] = df_processed['shipping_method'].apply(convert_shipping_method_to_list)

    # shipping_method 변수를 처리
    shipping_methods = ["온라인 전송", "직거래", "택배 거래"]
    for method in shipping_methods:
        df_processed[f'shipping_{method}'] = df_processed['shipping_method'].apply(lambda x: 1 if method in x else 0)
      
    # 원핫 인코딩 수행
    df_processed = pd.get_dummies(df_processed, columns=columns_to_encode)
    
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

# 이미지 개수 계산
def count_images_per_post(df, bucket_name):
    # Initialize S3 client
    s3 = boto3.client('s3')

    # Function to list objects in S3 bucket with prefix
    def list_images_in_s3(post_id, folder):
        prefix = f"{folder}/{post_id}_"
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if 'Contents' in response:
            return len(response['Contents'])
        return 0

    # Determine folder based on table
    def get_folder(table):
        if table == 'cellphone':
            return 'cellphone'
        elif table == 'tickets':
            return 'tickets'
        return ''

    # Calculate image count for each post
    df['image_count'] = df.apply(lambda row: list_images_in_s3(row['id'], get_folder(row['table'])), axis=1)
    return df

# 텍스트 전처리 함수
def preprocess_text(df):
    # 제목과 본문을 결합
    df['combined_text'] = df['title'] + ' ' + df['description']

    # 형태소 분석 및 토큰화
    def tokenize(text):
        return okt.morphs(text, stem=True)

    df['tokenized_text'] = df['combined_text'].apply(tokenize)
    
    return df

if __name__ == "__main__":
    # Query to fetch data from both tables
    query_cellphone = "SELECT *, 'cellphone' as table FROM cellphone ORDER BY id LIMIT 10"
    query_tickets = "SELECT *, 'tickets' as table FROM tickets ORDER BY id LIMIT 10"
    
    # Fetch data from cellphone table
    df_cellphone = pd.read_sql_query(query_cellphone, connectDB())
    
    # Fetch data from tickets table
    df_tickets = pd.read_sql_query(query_tickets, connectDB())
    
    # Combine data from both tables
    df_combined = pd.concat([df_cellphone, df_tickets], ignore_index=True)
    
    # Process the combined data
    df_processed = process_tabular_data(df_combined)
    
    # Set your S3 bucket name
    bucket_name = 'c2c-trade-image'
    
    # Count images per post
    df_processed = count_images_per_post(df_processed, bucket_name)

    # Preprocess text data
    df_processed = preprocess_text(df_processed)
        
    # Display the processed data
    print(df_processed[['id', 'image_count', 'tokenized_text']])
    
    print(df_processed.columns)