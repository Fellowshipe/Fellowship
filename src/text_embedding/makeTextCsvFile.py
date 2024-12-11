import psycopg2
import pandas as pd

# PostgreSQL 연결 설정
conn = psycopg2.connect(
    host="15.164.219.167",
    database="fellowship",
    user="postgres",
    password="9832"
)

# 데이터를 배치로 가져오기 위한 설정
# batch_size = 10000
# offset = 0
# total_rows = 30000

# 결과를 저장할 CSV 파일 이름
output_file = "cellphone_data_1007.csv"

# 컬럼 이름 추가 (utf-8 인코딩을 명시적으로 설정)
with open(output_file, mode='w', encoding='utf-8-sig') as f:
    f.write('id,cleaned_text\n')

    # SQL 쿼리: OFFSET과 LIMIT을 사용하여 데이터를 나눠서 가져오기
    query = f"SELECT id, cleaned_text FROM cellphone;"
    
    # 데이터 읽기
    df = pd.read_sql(query, conn)
    
    # 데이터를 CSV 파일로 추가 저장 (모드: 'a' = append, utf-8 인코딩)
    df.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
    
# 연결 종료
conn.close()
