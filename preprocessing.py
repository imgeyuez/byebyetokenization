"""
File which beholds various pre-processing steps.
"""

import pandas as pd 
from tqdm import tqdm
import re 
import random
from somajo import SoMaJo
import random
from nltk.corpus import wordnet
import copy 

random.seed(9)

def read_raw_data(datapath:str):
    """
    Function which reads in the csv file and 
    represents its input in a df.
    """

    df = pd.read_csv(datapath, delimiter=',', quotechar='"')

    return df


def read_processed_data(datapath:str):
    """
    Function which reads in the csv file and 
    represents its input in a df.
    """

    # Replace 'your_file.xlsx' with the path to your actual Excel file
    df = pd.read_excel(datapath)

    return df


def replace_values(df, column, value_to_replace, value_to_replace_with):
    """
    Function which replaces the values in the df.
    """
    df[column] = df[column].replace(value_to_replace, value_to_replace_with)

    return df


def remove_duplicates(df):
    
    # check for duplicates in 'text' column and count how many there are
    duplicates = df[df.duplicated(subset=['text'], keep=False)]  # All duplicates (keep=False)

    print(f"Total number of duplicate rows: {len(duplicates)/2}")

    # remove duplicates, keeping only the first occurrence
    df_cleaned = df.drop_duplicates(subset=['text'], keep='first')

    print(f"Number of rows after removing duplicates: {len(df_cleaned)} rows")

    return df_cleaned


def train_test_split(df, train_ratio=0.75):
    """
    Function which divides a given df 
    into train and test data (dfs) based on the
    defined ratio.
    """
    # split data into train and test data 
    total_rows = df.shape[0]
    train_size = int(total_rows*train_ratio)

    # shuffle df 
    df = df.sample(frac = 1)
    
    # split data into train and test data
    train = df[0:train_size]
    test = df[train_size:]

    return train, test 


def remove_other_languages(df, detector):
    """
    Function which removes all reviews
    that are in another language than English.
    """

    languages = dict()

    total_iterations = len(df)
    progressBar = tqdm(total=total_iterations, desc="Examine Languages...")

    for index, row in df.iterrows():
        text = row["text"]
        detected_language = detector.detect_language_of(text)

        if detected_language in languages.keys():
            languages[detected_language]["num"] += 1
            languages[detected_language]["texts"].append(row)
            
        else:
            languages[detected_language] = {
                "num": 1,
                "texts": [row]
            }
        
        progressBar.update(1)

    progressBar.close()


    # print overview of all languages included in the dataset 
    # for language in languages.keys():
        # print(f"Language: {language}\nReviews in that language: {languages[language]['num']}\n")
        # print(f"Language: {language},\nReviews in that language: {languages[language]["num"]}\n")

    keys = languages.keys()

    # just keep English reviews
    english_revs = languages[list(keys)[0]]["texts"]

    df_en = pd.DataFrame(english_revs)
    df_en = df_en.reset_index(drop=True)

    return df_en


def clean_reviews_duplicates(text):
    """
    Function which removes duplicates within reviews.
    Example, where duplicate is marked as -...-
    -Didn’t plan on making a review but am doing my part to protect this 
    masterpiece from review bombing. This game is wonderful, have played 
    for around 25 hours over the last few days and collecting resources 
    to upgrade the island and design your house is great. If you wanted 
    multiplayer then I’m sorry but Animal Crossing has always been mostly 
    single player, not sure what you were expecting.-Didn’t plan on making 
    a review but am doing my part to protect this masterpiece from review 
    bombing. This game is wonderful, have played for around 25 hours over 
    the last few days and collecting resources to upgrade the island and 
    design your house is great. If you wanted multiplayer then I’m sorry 
    but Animal Crossing has always been mostly single player, not sure what 
    you were expecting. Buy it, I beg you.… Expand
    """

    # take first 100 characters 
    beginning_of_rev = text[:100]

    # if the first 100 characters occur more than once, there's a duplication
    if text.count(beginning_of_rev) > 1:
        # use regex to find all occurrences of the duplicates
        pattern = re.escape(beginning_of_rev)  # scape to safely handle special characters
        
        # find start indices of all occurrences
        matches = [match.start() for match in re.finditer(pattern, text)]
        
        if len(matches) > 1:
            # second occurrence of the pattern
            second_occurrence_index = matches[1]

            # slice text to remove the duplicate part
            cleaned_text = text[second_occurrence_index:]

            return cleaned_text
        
        else:
            return text
    else:
        return text


def clean_reviews(text):
    """
    Function which cleans up other stuff in the reviews such as:
    - ... Expand
    - links
    - Disclaimer such as: 'This review contains spoilers, click expand to view.'
    """

    if not isinstance(text, str):  # Ensure the input is a string
        return text

    # remove "… Expand" at the end (use Unicode ellipsis or regular three dots)
    if "… Expand" in text:
        text = text.split("… Expand")[0]
    elif "... Expand" in text:  # Handle the case where three dots are used instead of a Unicode ellipsis
        text = text.split("... Expand")[0]

    # remove "This review contains spoilers" at the beginning
    if text.startswith("This review contains spoilers, click expand to view."):
        text = text.replace("This review contains spoilers, click expand to view.", "").strip()
    
    # remove potential links using regex
    text = re.sub(r'http\S+|www.\S+', '', text)  # Remove http://, https:// or www. links

    return text.strip()  # Ensure any extra spaces are removed


def sentence_tokenizer(tokenizer, text):

    s = tokenizer.tokenize_text([text])

    sentences = list()

    for sentence in s:
        
        sen = " ".join([token.text for token in sentence])
        sentences.append(sen)

    return sentences


def generate_review(tokenizer, text):
    
    sentences = sentence_tokenizer(tokenizer, text)
    boolean = [True, False]

    # make it random if sentence order should be shuffled or not
    shuffle = random.choice(boolean)
    delete = random.choice(boolean)
    lowercase = random.choice(boolean)
    exchange_words = random.choice(boolean)

    # if delete is true, delete 1 random sentence
    if delete and len(sentences) > 2:
        random_index = random.randint(0, len(sentences)-1)
        sentences.pop(random_index)
        
    # if shuffle:
    if shuffle:
        sentence_order = [i for i, sentence in enumerate(sentences)]
        random.shuffle(sentence_order)
        new_review = list()


        new_review = [sentences[num] for num in sentence_order]
        new_review = " ".join(new_review)
    else:
        new_review = " ".join(sentences)

    # exchange synonyms
    if exchange_words:
        new_review = synonym_replacement(new_review)

    # if lowercase is true, change everything in lowercase letters
    if lowercase:
        new_review = new_review.lower()

    return new_review


def synonym_replacement(review):
    """
    Perform synonym replacement on a review.
    Replace up to `n` words in the review with their synonyms.
    """

    # split the review into list of words
    words = review.split()
    
    # replace n number of words, based on the length of the review
    if len(words) > 30:
        n = 5
    elif len(words) > 10:
        n = 2
    else:
        return review

    # create copy of words
    new_review = copy.deepcopy(words)

    # create list of words in the review that have available synonyms in WordNet
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    
    # shuffle list of words randomly to select different words for replacement each time
    random.shuffle(random_word_list)

    # initialize counter for number of words that have been replaced
    num_replaced = 0
    
    # loop through the words with synonyms in the random order
    for random_word in random_word_list:
        # get all synonym sets (synsets) for the current word from WordNet
        synonyms = wordnet.synsets(random_word)
        
        # if synonyms exist for the word, perform the replacement
        if synonyms:
            # choose the first synonym lemma from the first synset
            random.shuffle(synonyms)
            synonym = synonyms[0].lemmas()[0].name()
            
            # replace the word in the new_review with its synonym

            for j, word in enumerate(words):
                if word == random_word and word != synonym:
                    new_review[j] = synonym
                    # increase the number of words replaced
                    num_replaced += 1

                else:
                    pass
            
        # If required number of replacements (n) has been done, break
        if num_replaced >= n:
            break

    # join the modified words back into string
    new_review = " ".join(new_review)
    return new_review    


def data_augmentation(df):

    augmented_df = df.copy(deep=True)

    total_iterations = len(df)
    progressBar = tqdm(total=total_iterations, desc="Generate new data...")

    tokenizer = SoMaJo(language="en_PTB")

    for ind, row in df.iterrows():
        # get ranking
        row_items = row.tolist()
        grade = row["grade"]
        text = row["text"]

        """
            if grade == 0: don't add anything,        1412 (1412)-> add nothing
            elif grade == 1: double the data,         456  (228) -> add 1 new rev per revs 
            elif grade == 2: triple the data,         549  (183) ->  add 2 new revs per rev 
            elif grade == 3: triple the data,         312  (78)  -> add 3 new revs per rev
            elif grade == 4: double the data ,        688  (344) -> add 1 new revs per rev
            elif grade == 5: double the data          1502 (751) -> add one new rev per rev
        """

        if grade == 0:
            progressBar.update(1)
            continue

        else:
            # 1. split text into sentences with perhaps nltk
            # 2. change order of sentences with random 
            # 2.5 maybe even delete some
            # 3. exchange up to 4 words with synonyms 
            # 4. make some in complete lowercase 

            counter = 0

            if grade == 1 or grade == 4 or grade == 5:

                new_review = generate_review(tokenizer, text)
                new_row = copy.deepcopy(row_items)
                new_row[2] = new_review

                # add new review to the df
                augmented_df.loc[len(augmented_df)] = new_row

            elif grade == 2:
                while counter < 2:
                    new_review = generate_review(tokenizer, text)
                    new_row = copy.deepcopy(row_items)
                    new_row[2] = new_review

                    # add new review to the df
                    augmented_df.loc[len(augmented_df)] = new_row

                    counter += 1

            elif grade == 3:
                while counter < 3:
                    new_review = generate_review(tokenizer, text)
                    new_row = copy.deepcopy(row_items)
                    new_row[2] = new_review

                    # add new review to the df
                    augmented_df.loc[len(augmented_df)] = new_row

                    counter += 1

        progressBar.update(1)
    progressBar.close()

    return augmented_df


def remove_stopwords(tokens:list, stopwords: list) -> list:
    """
        Function which takes in the tokenized reviews
        and removes all tokens that are definded 
        as stopwords 
    """

    # make tokens lowercase 
    lowercased_tokens = [token.lower() for token in tokens]

    # for each token check if it is a stopword
    # & append all not-stopword tokens to new list 
    # of tokens
    cleaned_review_tokens = list()
    for index, token in enumerate(lowercased_tokens):
        if token not in stopwords:
            cleaned_review_tokens.append(tokens[index])

    return cleaned_review_tokens


        


