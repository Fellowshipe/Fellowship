# Basic
import pandas as pd
from tqdm import tqdm
import numpy as np
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


def Tokenize(data, model_ckpt, batch_size, output_csv, pooling='mean'): 
    """
    model_ckpt: klue/bert-base
    batch_size: recommend that the value of this variable be 2 or 4
    output_csv: the path to save the result CSV file
    pooling: 'mean' or 'max' for the pooling method
    """
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = AutoModel.from_pretrained(model_ckpt, output_hidden_states=True).to(device)

    embeddings = []
    ids = []
    text_list = data['cleaned_text'].tolist()
    id_list = data['id'].tolist()
    token_length_list = data['token_length'].tolist()

    for i in tqdm(range(0, len(text_list), batch_size)):
        batch_texts = text_list[i:i+batch_size]
        batch_ids = id_list[i:i+batch_size]
        batch_token_lengths = token_length_list[i:i+batch_size]

        for j, text in enumerate(batch_texts):
            if batch_token_lengths[j] > 512:
                embeddings.append([])
                ids.append(batch_ids[j])
            else:
                inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)

                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # 마지막 4개 레이어의 hidden states 추출
                hidden_states = outputs.hidden_states[-4:]
                cls_embeddings = [layer[:, 0, :].cpu().numpy() for layer in hidden_states]
                
                # pooling 방식 적용
                if pooling == 'mean':
                    cls_embedding = np.mean(cls_embeddings, axis=0).flatten().tolist()
                elif pooling == 'max':
                    cls_embedding = np.max(cls_embeddings, axis=0).flatten().tolist()
                else:
                    raise ValueError("Pooling method should be either 'mean' or 'max'")

                embeddings.append(cls_embedding)
                ids.append(batch_ids[j])

    result_df = pd.DataFrame({'id': ids, 'embedding': embeddings})
    result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    return result_df


batch_size = 2


df = pd.read_csv("/Users/wnsgud/workplace/Fellowship/numberToken.csv")
df = df[:1]

# 내가 생각한 것은 (데이터프레임(텍스트))
text_extraction = Tokenize(df, "klue/bert-base", batch_size=2, output_csv="embedding_text.csv")
#print(len(text_extraction[0])) # 768차원
print(text_extraction)