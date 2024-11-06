from kobert_tokenizer import KoBERTTokenizer

class KoBERT_Tokenizer:
    def __init__(self):
        self.koBERT = KoBERTTokenizer

    def tokenizer(self, text):
         tokenizer = self.koBERT.from_pretrained('skt/kobert-base-v1')
         tokened = tokenizer.encode(text)
         return tokened