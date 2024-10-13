"""
File which defines the tokenizer based on white-space
token separation.
"""

def tokenizer_whitespace(text):
    """
    Function which tokenizes a text based on the
    split on whit-space approach.
    """

    tokenized = text.split(" ")
    return tokenized

