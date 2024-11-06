import torch
from transformers import BertModel

class ExtractFeature:
    def __init__(self, tokenize=KoBERT_Tokenizer):
        self.tokenize = tokenize
        self.kobert_model = BertModel.from_pretrained('skt/kobert-base-v1')

    def extract_feature(self, text):
        encoded_input = 

        input_ids = torch.tensor([])