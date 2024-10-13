"""
File which trains the bpe tokenizer
"""

import re
from collections import defaultdict
from tqdm import tqdm

def get_stats(vocab):
    """
    Given a vocabulary (dictionary mapping words to frequency counts), returns a 
    dictionary of tuples representing the frequency count of pairs of characters 
    in the vocabulary.
    """
    pairs = defaultdict(int)
    # for each key, value pair in vocab
    for word, freq in vocab.items():
        # split word into characters
        symbols = word.split()
        # count frequency of each pair of characters
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    """
    Given a pair of characters and a vocabulary, returns a new vocabulary with the 
    pair of characters merged together wherever they appear.
    """
    # initialize modified vocabulary
    v_out = {}
    # define bigram
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def get_vocab(data):
    """
    Given a dataset, returns a dictionary of words mapping to their frequency 
    count in the data.
    """
    vocab = defaultdict(int)
    for line in data:
        for word in line.split():
            vocab[' '.join(list(word)) + ' </w>'] += 1
    return vocab

def learn_bpe(df, n=2000):
    """
    Learn BPE vocabulary and return the merge rules learned from the training data.
    """

    # get all reviews from the df 
    reviews = df["text"].tolist()

    corpus = " ".join(reviews)

    data = corpus.split('.')

    # Initialize vocabulary from the data
    vocab = get_vocab(data)
    
    # List to store merge rules
    merge_rules = []

    total_iterations = n
    progressBar = tqdm(total=total_iterations, desc="Train BPE...")

    for i in range(n):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        
        # Store the best pair (merge rule) for future application
        merge_rules.append(best)
        
        # Merge the best pair in the vocabulary
        vocab = merge_vocab(best, vocab)

        progressBar.update(1)

    progressBar.close()

    proper_vocab = list()

    for v in vocab:
        v_s = v.split()
        for e in v_s:
            if e != "":
                proper_vocab.append(e)
    
    # Return the final vocabulary and merge rules
    return proper_vocab, merge_rules


def apply_bpe(word, merge_rules):
    """
    Apply the learned BPE merges to a given word.
    """
    # Split the word into characters with </w> at the end
    word = list(word) + ['</w>']
    
    # Iteratively apply the merge rules
    for pair in merge_rules:
        i = 0
        while i < len(word) - 1:
            # Check if the current character pair matches the merge rule
            if (word[i], word[i+1]) == pair:
                # Merge the pair
                word = word[:i] + [''.join(pair)] + word[i+2:]
            else:
                i += 1
                
    return word


def tokenizer_bpe(review, merge_rules):
    """
    Function which applies the trained BPE
    on new sentences.
    """

    tokenized_review = list()

    sentences = review.split(".")

    for sentence in sentences:
        tokenized_sentence = []

        for word in sentence.split():
            tokenized_word = apply_bpe(word, merge_rules)
            last_token = tokenized_word[-1].replace("</w>", "")
            tokenized_word[-1] = last_token
            tokenized_sentence.extend(tokenized_word)  # join subwords with spaces for readability

        tokenized_review.extend(tokenized_sentence)

    return tokenized_review
    print("Tokenized Sentence:", tokenized_sentence)

