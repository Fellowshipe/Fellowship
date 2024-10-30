import pandas as pd
import numpy as np

text_df = pd.read_csv("../embedding_text.csv")
image_df = pd.read_csv("../swin_image_embeddings_partial.csv", header=None)
tabular_df = pd.read_csv("../tabular_data.csv")

# 이미지 처리
# 첫 번째 컬럼을 'id', 나머지 컬럼을 'embedding_1', 'embedding_2', ..., 'embedding_n' 형식으로 이름 지정
image_df.columns = ['id'] + [f'image_embedding_{i}' for i in range(1, image_df.shape[1])]

# id를 기준으로 groupby하여 각 id에 대한 임베딩의 평균값을 계산
image_df = image_df.groupby('id').mean().reset_index()

# 텍스트 처리
# 텍스트 임베딩 열을 리스트로 변환
text_df['embedding'] = text_df['embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=', '))

# 임베딩 벡터를 개별 컬럼으로 풀어주고 컬럼명 설정
embedding_columns = pd.DataFrame(text_df['embedding'].tolist(), index=text_df.index)
embedding_columns.columns = [f'text_embedding_{i}' for i in range(embedding_columns.shape[1])]

# 'id'와 개별 임베딩 컬럼을 결합
df_text_embedding_expanded = pd.concat([text_df['id'], embedding_columns], axis=1)

# 태뷸라 처리
tabular_df['id'] = text_df['id']

print(tabular_df.shape)
print(df_text_embedding_expanded.shape)
print(image_df.shape)

# ID를 기준으로 이미지와 텍스트 임베딩을 결합
df_merged = pd.merge(tabular_df, df_text_embedding_expanded, on='id', how='inner')
df_merged = pd.merge(df_merged, image_df, on='id', how='inner')

# 결합된 임베딩 확인
print(df_merged)

df_merged.to_csv("tabular_text_image_data.csv", index=False)
tabular_df.drop(columns='is_fraud').to_csv("tabular_data.csv", index=False)
df_text_embedding_expanded.to_csv("text_data.csv", index=False)
image_df.to_csv("image_data.csv", index=False)
tabular_df[['id', 'is_fraud']].to_csv("class_data.csv", index=False)