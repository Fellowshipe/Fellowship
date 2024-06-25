import pandas as pd
import string
import boto3
from konlpy.tag import Okt
import os
import torch
import clip  # OpenAI의 CLIP 라이브러리 가져오기
from PIL import Image
from tqdm import tqdm
from dbControl.connect_db import connectDB

# 텍스트 전처리 함수
def preprocess_text(df):
    # 제목과 본문을 결합
    df['combined_text'] = df['title'] + ' ' + df['description']
    return df

# 이미지 연결 및 URL 수집 함수
def count_images_per_post(df, bucket_name):
    # Initialize S3 client
    s3 = boto3.client('s3')

    # Function to list objects in S3 bucket with prefix and get URLs
    def list_images_in_s3(post_id, folder):
        prefix = f"{folder}/{post_id}_"
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if 'Contents' in response:
            image_urls = [item['Key'] for item in response['Contents']]
            return len(response['Contents']), image_urls
        return 0, []

    # Determine folder based on table
    def get_folder(table):
        if table == 'cellphone':
            return 'cellphone'
        elif table == 'tickets':
            return 'tickets'
        elif table == 'clothes':
            return 'clothes'
        return ''

    # Calculate image count and get URLs for each post
    df['image_count'], df['image_urls'] = zip(*df.apply(lambda row: list_images_in_s3(row['id'], get_folder(row['table'])), axis=1))
    return df

# Function to fetch image from S3
def fetch_image_from_s3(bucket_name, image_key):
    s3 = boto3.client('s3')
    try:
        response = s3.get_object(Bucket=bucket_name, Key=image_key)
        image = Image.open(response['Body']).convert("RGB")
        return image
    except s3.exceptions.NoSuchKey:
        print(f"No such key: {image_key}")
        return None

# Function to process images and texts
def process_images_and_texts(df, bucket_name, device):
    texts = df['combined_text'].tolist()
    image_keys = df['image_urls'].tolist()

    texts_expanded = []
    images = []

    for text, keys in zip(texts, image_keys):
        for image_key in keys:
            print(f"Trying to fetch image with key: {image_key}")  # Debugging line
            image = fetch_image_from_s3(bucket_name, image_key)
            if image:
                image = preprocess(image).unsqueeze(0).to(device)
                images.append(image)
                texts_expanded.append(text)
            else:
                images.append(torch.zeros((1, 3, 224, 224)).to(device))
                texts_expanded.append(text)

    return texts_expanded, images

# Main function to load data, preprocess and evaluate with CLIP
if __name__ == "__main__":
    # Query to fetch data from all tables
    query_cellphone_fraud = "SELECT *, 'cellphone' as table FROM cellphone WHERE is_fraud = True ORDER BY id LIMIT 10"
    query_tickets_fraud = "SELECT *, 'tickets' as table FROM tickets WHERE is_fraud = True ORDER BY id LIMIT 10"
    query_clothes_fraud = "SELECT *, 'clothes' as table FROM clothes WHERE is_fraud = True ORDER BY id LIMIT 10"
    query_cellphone_nonfraud = "SELECT *, 'cellphone' as table FROM cellphone WHERE is_fraud = False ORDER BY id LIMIT 100"
    query_tickets_nonfraud = "SELECT *, 'tickets' as table FROM tickets WHERE is_fraud = False ORDER BY id LIMIT 100"
    query_clothes_nonfraud = "SELECT *, 'clothes' as table FROM clothes WHERE is_fraud = False ORDER BY id LIMIT 100"   
     
    # Fetch data from table
    df_cellphone_fraud = pd.read_sql_query(query_cellphone_fraud, connectDB())
    df_tickets_fraud = pd.read_sql_query(query_tickets_fraud, connectDB())
    df_clothes_fraud = pd.read_sql_query(query_clothes_fraud, connectDB())
    df_cellphone_nonfraud = pd.read_sql_query(query_cellphone_nonfraud, connectDB())
    df_tickets_nonfraud = pd.read_sql_query(query_tickets_nonfraud, connectDB())
    df_clothes_nonfraud = pd.read_sql_query(query_clothes_nonfraud, connectDB())
    
    # Combine data from all tables
    df_combined = pd.concat([df_cellphone_fraud, df_tickets_fraud, df_clothes_fraud, df_cellphone_nonfraud, df_tickets_nonfraud, df_clothes_nonfraud], ignore_index=True)
    
    # Set your S3 bucket name
    bucket_name = 'c2c-trade-image'
    
    # Preprocess text data
    df_processed = preprocess_text(df_combined)
    
    # Count images per post and get URLs
    df_processed = count_images_per_post(df_processed, bucket_name)
    
    # Load the CLIP model and preprocessing method
    model, preprocess = clip.load("ViT-B/32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Process images and texts
    texts, images = process_images_and_texts(df_processed, bucket_name, device)

    # Tokenize texts
    text_tokens = clip.tokenize(texts, truncate=True).to(device)

    # Compute features
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        image_features = torch.cat(images)
        image_features = model.encode_image(image_features)

    # Normalize features
    text_features /= text_features.norm(dim=-1, keepdim=True)
    image_features /= image_features.norm(dim=-1, keepdim=True)

    # Compute similarity
    similarity = (text_features @ image_features.T).cpu().numpy()

    # Compute average similarity per post
    similarities = []
    start = 0
    for count in df_processed['image_count']:
        end = start + count
        post_similarities = similarity[start:end, start:end]
        average_similarity = post_similarities.diagonal().mean()
        similarities.append(average_similarity)
        start = end

    # Evaluate the model: Here we assume high similarity implies non-fraud and low similarity implies fraud
    df_processed['similarity'] = similarities
    df_processed['is_fraud_pred'] = df_processed['similarity'].apply(lambda x: 1 if x < 0.5 else 0)

    # Compare with actual fraud labels
    accuracy = (df_processed['is_fraud_pred'] == df_processed['is_fraud']).mean()
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    # Display the processed data with predictions
    print(df_processed[['id', 'table', 'is_fraud', 'is_fraud_pred', 'similarity']])
