"""
File which uses rule-based tokenization trained
on data suitable for the task.
The trained model used is SoMaJo.
"""

from somajo import SoMaJo

def tokenizer_rule_based(text, tokenizer):
    s = tokenizer.tokenize_text([text])

    tokenized_review = list()

    for sentence in s:
        sen = [token.text for token in sentence]
        tokenized_review.extend(sen)

    return tokenized_review



