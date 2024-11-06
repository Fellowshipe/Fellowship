# Basic
import pandas as pd

# KLUE_BERT
from transformers import AutoTokenizer

def add_token_length_column_and_save_csv(data, model_ckpt, output_csv):
    """
    data: a DataFrame containing 'id' and 'cleaned_text' columns
    model_ckpt: the model checkpoint for the tokenizer (e.g., 'klue/bert-base')
    output_csv: the path to save the CSV file (e.g., 'numberToken.csv')
    """
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    # 'cleaned_text'의 토큰 길이를 저장할 리스트
    token_lengths = []

    # 각 텍스트의 토큰 길이를 계산
    for text in data['cleaned_combined_text']:
        tokens = tokenizer(text, truncation=False, padding=False)  # 토큰화
        token_lengths.append(len(tokens['input_ids']))  # 토큰 길이 추가

    # 데이터프레임에 새로운 열 추가
    data['token_length'] = token_lengths

    # 데이터프레임을 CSV 파일로 저장
    data.to_csv(output_csv, index=False)

    return data

# 파일의 인코딩 감지
# with open("/Users/wnsgud/workplace/Fellowship/cellphone_data_1007.csv", 'rb') as f:
#     result = chardet.detect(f.read(10000))  # 처음 10,000바이트만 읽어서 인코딩을 감지
#     print(f"Detected encoding: {result['encoding']}")


# 사용 예시
df = pd.read_csv("/Users/wnsgud/workplace/Fellowship/cellphone_text.csv")
df.info()

model_ckpt = 'klue/bert-base'
output_csv = 'numberToken1007.csv'

# 토큰 길이 열 추가 및 CSV 파일 저장
updated_data = add_token_length_column_and_save_csv(df, model_ckpt, output_csv)
# 토큰 길이가 512개를 초과하는 행만 필터링
over_512_tokens = updated_data[updated_data['token_length'] > 512]
    
# 결과 출력
over_512_tokens.info()