import requests
import boto3

from dotenv import load_dotenv
import os

def download_image(url):
    """주어진 URL에서 이미지를 다운로드하고 바이트 데이터를 반환합니다."""
    response = requests.get(url)
    response.raise_for_status()  # 4xx 또는 5xx 응답을 확인하고 예외를 발생시킵니다.
    return response.content

def upload_to_s3(bucket_name, s3_path, image_bytes):
    try:
        load_dotenv()

        key_id=os.getenv('AWS_ACCESS_KEY_ID')
        access_key=os.getenv('AWS_SECRET_ACCESS_KEY')

        """이미지 바이트 데이터를 S3의 지정된 경로에 업로드합니다."""
        s3_client = boto3(
            's3',  
            aws_access_key_id = key_id,
            aws_secret_access_key = access_key
        )
        s3_client.upload_file(
            s3_path,
            bucket_name,
            image_bytes
        )
    except Exception as e:
        print(f"Upload Error occurred: {e}")