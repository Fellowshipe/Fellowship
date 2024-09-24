import pandas as pd
import string
import boto3
import os
import torch
import clip  # OpenAI의 CLIP 라이브러리 가져오기
from PIL import Image
from tqdm import tqdm
import json
import requests

# Load texts and images URLs from JSON
def load_texts_images_from_json(json_file='text_image_pairs.json'):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data
    texts = [item['text'] for item in data]
    image_urls = [item['image_urls'] for item in data]
    return texts, image_urls


# Main function to load data, preprocess and evaluate with CLIP
if __name__ == "__main__":
    
    json_file = 'text_image_pairs.json'
    
    texts, image_urls = load_texts_images_from_json(json_file)
    
    print(texts[0], image_urls[0])