# Basic
import pandas as pd
from tqdm import tqdm

# KLUE_BERT
from transformers import AutoTokenizer, AutoModel
import torch

# def Tokenize(data, model_ckpt, batch_size): # function of extracting [CLS] Token embedding from BERT-based model

#     """
#     model_ckpt: klue/bert-base
#     col_name: append cls token embedding data column into dataframe
#     batch_size: recommend that the value of this variable be 2 or 4
#     """

#     tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(device)
#     model = AutoModel.from_pretrained(model_ckpt).to(device)

#     embeddings = []
#     text_list = data['cleaned_text'].tolist()

#     for i in tqdm(range(0, len(text_list), batch_size)):
#         batch_texts = text_list[i:i+batch_size]

#         inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True) # default of max_length is 512
#         input_ids = inputs["input_ids"].to(device)
#         attention_mask = inputs["attention_mask"].to(device)

#         with torch.no_grad():
#             embedding = model(input_ids=input_ids, attention_mask=attention_mask)
#         embeddings.append(embedding.last_hidden_state[:, 0, :])  # append CLS token embedding data

#     # Stack embeddings into a tensor
#     stacked_embeddings = torch.cat(embeddings, dim=0)

#     stacked_embeddings = stacked_embeddings.cpu().numpy()

#     result = stacked_embeddings.tolist()

#     return result


def Tokenize(data, model_ckpt, batch_size, output_csv): 
    """
    model_ckpt: klue/bert-base
    batch_size: recommend that the value of this variable be 2 or 4
    output_csv: the path to save the result CSV file
    """
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = AutoModel.from_pretrained(model_ckpt).to(device)

    embeddings = []
    ids = []
    text_list = data['cleaned_text'].tolist()
    id_list = data['id'].tolist()
    token_length_list = data['token_length'].tolist()  # token_length 열에서 토큰 길이를 가져옴

    for i in tqdm(range(0, len(text_list), batch_size)):
        batch_texts = text_list[i:i+batch_size]
        batch_ids = id_list[i:i+batch_size]
        batch_token_lengths = token_length_list[i:i+batch_size]

        for j, text in enumerate(batch_texts):
            # token_length가 512를 초과하면 임베딩을 진행하지 않음
            if batch_token_lengths[j] > 512:
                embeddings.append([])  # 512를 넘는 경우 빈 리스트 추가
                ids.append(batch_ids[j])  # 해당 id도 추가
            else:
                # 토큰화 및 임베딩 진행
                inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)

                with torch.no_grad():
                    embedding = model(input_ids=input_ids, attention_mask=attention_mask)
                
                last_four_layers = embedding.hidden_states[-4:]  # list of 4 tensors

                # 각 층에 대해 global average pooling
                pooled_layers = []
                for layer in last_four_layers:
                    # layer shape: (batch_size, seq_length, hidden_size)
                    pooled = layer.mean(dim=1)  # average over seq_length → shape: (batch_size, hidden_size)
                    pooled_layers.append(pooled)
                
                # 4개 층 평균 → shape: (batch_size, hidden_size)
                avg_embedding = torch.stack(pooled_layers, dim=0).mean(dim=0)  
                
                # numpy 변환 및 저장
                for j in range(avg_embedding.shape[0]):
                    embedding_vec = avg_embedding[j].cpu().numpy().flatten().tolist()
                    embeddings.append(embedding_vec)
                    ids.append(batch_ids[j])


    # DataFrame으로 id와 임베딩 결과를 저장
    result_df = pd.DataFrame({'id': ids, 'embedding': embeddings})

    # 결과를 CSV 파일로 저장
    result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    return result_df


batch_size = 2


df = pd.read_csv("/Users/wnsgud/workplace/Fellowship/numberToken.csv")
df = df[:1000]

# 내가 생각한 것은 (데이터프레임(텍스트))
text_extraction = Tokenize(df, "klue/bert-base", batch_size=2, output_csv="embedding_text.csv")
#print(len(text_extraction[0])) # 768차원
print(text_extraction)
