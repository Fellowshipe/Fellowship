import pandas as pd
import string
import boto3
from konlpy.tag import Okt
import os
import torch
from PIL import Image
from tqdm import tqdm
import json
import requests
from dbControl.connect_db import connectDB
from transformers import CLIPProcessor, CLIPModel

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
            image_urls = [f"https://{bucket_name}.s3.amazonaws.com/{item['Key']}" for item in response['Contents']]
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

# Function to fetch image from URL
def fetch_image_from_url(image_url):
    try:
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        return image
    except Exception as e:
        print(f"Error fetching image from {image_url}: {e}")
        return None

# Save texts and images URLs as JSON
def save_texts_images_as_json(df, json_file='text_image_pairs.json'):
    data = [{'text': row['combined_text'], 'image_urls': row['image_urls']} for _, row in df.iterrows()]
    with open(json_file, 'w') as f:
        json.dump(data, f)

# Load texts and images URLs from JSON
def load_texts_images_from_json(json_file='text_image_pairs.json'):
    with open(json_file, 'r') as f:
        data = json.load(f)
    texts = [item['text'] for item in data]
    image_urls = [item['image_urls'] for item in data]
    return texts, image_urls

# Main function to load data, preprocess and evaluate with CLIP
if __name__ == "__main__":
    # Set your S3 bucket name
    bucket_name = 'c2c-trade-image'
    processed_data_file = 'processed_data.csv'
    
    if not os.path.exists(processed_data_file):
        # Query to fetch data from all tables
        query_cellphone_fraud = "SELECT *, 'cellphone' as table FROM cellphone WHERE is_fraud = True ORDER BY id LIMIT 25"
        query_tickets_fraud = "SELECT *, 'tickets' as table FROM tickets WHERE is_fraud = True ORDER BY id LIMIT 25"
        query_cellphone_nonfraud = "SELECT *, 'cellphone' as table FROM cellphone WHERE is_fraud = False ORDER BY id LIMIT 25"
        query_tickets_nonfraud = "SELECT *, 'tickets' as table FROM tickets WHERE is_fraud = False ORDER BY id LIMIT 25"

        # Fetch data from table
        df_cellphone_fraud = pd.read_sql_query(query_cellphone_fraud, connectDB())
        df_tickets_fraud = pd.read_sql_query(query_tickets_fraud, connectDB())
        df_cellphone_nonfraud = pd.read_sql_query(query_cellphone_nonfraud, connectDB())
        df_tickets_nonfraud = pd.read_sql_query(query_tickets_nonfraud, connectDB())

        # Combine data from all tables
        df_combined = pd.concat([df_cellphone_fraud, df_tickets_fraud, df_cellphone_nonfraud, df_tickets_nonfraud], ignore_index=True)
        
        # Preprocess text data
        df_processed = preprocess_text(df_combined)
        
        # Count images per post and get URLs
        df_processed = count_images_per_post(df_processed, bucket_name)
        
        # Save processed data to CSV
        df_processed.to_csv(processed_data_file, index=False)
    else:
        # Load processed data from CSV
        df_processed = pd.read_csv(processed_data_file)
       
    # Load the CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

        
    # Fetch images from URLs
    images = []
    texts_expanded = []
    for text, urls in tqdm(zip(df_processed['combined_text'], df_processed['image_urls'])):
        urls = eval(urls)  # Convert string representation of list back to list
        for url in urls:
            image = fetch_image_from_url(url)
            if image:
                images.append(image)
                texts_expanded.append(text)
            else:
                images.append(Image.new("RGB", (224, 224)))  # 빈 이미지를 대신 사용


    # Process texts and images
    inputs = processor(text=texts_expanded, images=images, return_tensors="pt", padding=True, truncation=True).to(device)

    # Compute features
    with torch.no_grad():
        outputs = model(**inputs)
        text_features = outputs.text_embeds
        image_features = outputs.image_embeds

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
