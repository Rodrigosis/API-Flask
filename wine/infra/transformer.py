from transformers import AutoTokenizer


class Transformer:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(self, text: str):
        tkn = self.tokenizer(text, return_tensors="np")
        return tkn['input_ids']
