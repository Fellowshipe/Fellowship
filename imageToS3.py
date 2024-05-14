import requests
from boto3.session import Session

def download_image(url):
    """주어진 URL에서 이미지를 다운로드하고 바이트 데이터를 반환합니다."""
    response = requests.get(url)
    response.raise_for_status()  # 4xx 또는 5xx 응답을 확인하고 예외를 발생시킵니다.
    return response.content

def upload_to_s3(bucket_name, s3_path, image_bytes):
    """이미지 바이트 데이터를 S3의 지정된 경로에 업로드합니다."""
    session = Session()  # AWS 자격 증명과 구성을 사용하여 세션 생성
    s3 = session.resource('s3')
    s3_object = s3.Object(bucket_name, s3_path)
    s3_object.put(Body=image_bytes)  # S3에 이미지 데이터 업로드