import pandas as pd
from transformers import BertTokenizer

# Define a function to tokenize text and return the actual tokens
def tokenizer_bert(text, tokenizer):
    # Tokenize the text into subword tokens
    tokens = tokenizer.tokenize(text)
    return tokens



