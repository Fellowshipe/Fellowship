import pandas as pd

def split_and_save_dataframe(data, batch_size, base_filename):
    """
    data: 입력 데이터프레임
    batch_size: 한 파일에 저장할 행의 개수 (예: 15000)
    base_filename: 파일 이름의 기본 값 (각 파일에 번호가 붙음)
    """
    num_batches = (len(data) // batch_size) + 1

    for i in range(num_batches):
        # 데이터프레임의 각 배치에 대해 시작과 끝 인덱스 계산
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        # 데이터프레임의 일부분 추출
        batch_data = data.iloc[start_idx:end_idx]

        # 각 파일에 저장할 파일명 생성
        output_filename = f"{base_filename}_part_{i+1}.csv"

        # 파일 저장
        batch_data.to_csv(output_filename, index=False, encoding='utf-8-sig')

        print(f"Saved {output_filename} with {len(batch_data)} rows.")

# 사용 예시
data = pd.read_csv("/Users/wnsgud/workplace/Fellowship/embedding_text.csv")

# 15000개 단위로 파일을 나눠 저장
split_and_save_dataframe(data, batch_size=15000, base_filename='divide_text_embedding')
