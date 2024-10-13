"""
File with various utilization functions
needed within the code that don't find place
in any of the other files :)
"""

from tokenizer.whitespace import *
import torch
from torch.utils.data import DataLoader, RandomSampler

def get_vocababulary(df, column_name):
    """
    Function which iterates through all tokenized
    reviews in the column and adds them to the vocabulary.
    """
    vocab = set()
    for i, row in df.iterrows():
        tokens = row[column_name]
        for token in tokens:
            vocab.add(token)
    return vocab

def unknown_tokens(tokenizer, vocab, bert_vocab):
    print(tokenizer)
    unknown = 0
    for word in list(set(vocab)):
        w = word.replace("</w>", "") 
        if w not in bert_vocab:
            unknown += 1
    return unknown 


def encode_data(df, tokenizer, bert_tokenizer):
    """
    Function which takes a df as input
    and returns a list of encoded tokens, 
    representing the input text
    """

    encoded_reviews = list()
    attention_masks = list()

    cls_token_id = bert_tokenizer.cls_token_id  # [CLS] token ID
    sep_token_id = bert_tokenizer.sep_token_id  # [SEP] token ID

    counter = 0

    for index, row in df.iterrows():
        tokenized_text = row[f"tokenized_{tokenizer}"]
        token_ids = [bert_tokenizer.convert_tokens_to_ids(token) for token in tokenized_text]

        # if num token exceeds 512 limit, "leave out thee middle part"
        if len(token_ids) > 510:
            part1 = token_ids[:255]
            part2 = token_ids[len(token_ids)-255:]

            token_ids = part1 + part2

            counter += 1


        # add the special tokens for classification (CLS) and 
        # end of sequence marker (SEP)
        token_ids = [cls_token_id] + token_ids + [sep_token_id]

        # Create attention masks (1 for real tokens, 0 for padding)
        attention_mask = [1 for ids in token_ids]

        # padd the sequences to a length of 512
        while len(token_ids) < 512:
            # append padding ID
            token_ids.append(0)
            attention_mask.append(0)


        encoded_reviews.append(token_ids)
        attention_masks.append(attention_mask)


    return counter, torch.IntTensor(encoded_reviews), torch.IntTensor(attention_masks)


def data_loader(data, batch_size=8):
    dataloader = DataLoader(
            data,  # The training samples.
            sampler = RandomSampler(data), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )
    
    return dataloader

def count_unknown(train_ids):
    unk_token_counter = 0
    for review in train_ids:
        unk_token_counter += torch.sum(review == 100).item()
    return unk_token_counter