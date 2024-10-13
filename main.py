from preprocessing import *
import openpyxl

from tokenizer.whitespace import *
from tokenizer.bpe import *
from tokenizer.rule_based import *
from tokenizer.bert import *
from utils import *

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split, RandomSampler, SequentialSampler
import transformers
from transformers import BertConfig, BertForSequenceClassification, AdamW, BertConfig,BertTokenizer,get_linear_schedule_with_warmup

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

from lingua import LanguageDetectorBuilder
import json


def prepare_train_data(DATAPATH):

    # read in data 
    df = read_raw_data(DATAPATH)
    # print(df.head)

    grade_counts = df['grade'].value_counts()
    print(grade_counts)


    #########################################
    ###########REMOVE DUBLICATES#############

    df = remove_duplicates(df)

    #########################################
    #########MODIFY CLASS LABELS ############

    # get overview of freq of rankings 
    grade_counts = df['grade'].value_counts()
    print(grade_counts)

    # replace rankings 
    """
        as such:
        0, 1 -> 0
        2, 3 -> 1
        4, 5 -> 2
        6, 7 -> 3
        8, 9 -> 4
        10 -> 5
    """

    df = replace_values(df, "grade", 1, 0)
    df = replace_values(df, "grade", 2, 1)
    df = replace_values(df, "grade", 3, 1)
    df = replace_values(df, "grade", 4, 2)
    df = replace_values(df, "grade", 5, 2)
    df = replace_values(df, "grade", 6, 3)
    df = replace_values(df, "grade", 7, 3)
    df = replace_values(df, "grade", 8, 4)
    df = replace_values(df, "grade", 9, 4)
    df = replace_values(df, "grade", 10, 5)

    # overview of new counts
    grade_counts = df['grade'].value_counts()
    print(grade_counts)

    # divide dataset into train and test 
    train_data, test_data = train_test_split(df)

    print(train_data.info())
    print(test_data.info())


    #########################################
    #########CLEAN DATASET(s)################
    # keep only English reviews 
    df_en_train = remove_other_languages(train_data, language_detector)
    # df_en_test = remove_other_languages(test_data)

    # clean sentence (duplicates)
    df_en_train['text'] = df_en_train['text'].progress_apply(clean_reviews_duplicates)

    # apply to "text" column
    df_en_train['text'] = df_en_train['text'].progress_apply(clean_reviews)



    #########################################
    ###########DATA AUGMENTATION#############

    # overview of new counts
    print(df_en_train.info())
    grade_counts = df_en_train['grade'].value_counts()
    print(grade_counts)

    mod_train_data = data_augmentation(df_en_train)
    print(mod_train_data.info())

    grade_counts = mod_train_data['grade'].value_counts()
    print(grade_counts)

    train_file_name = 'train_data.xlsx'
    test_file_name = 'test_data.xlsx'

    # save datasets within excel to check train data only 
    mod_train_data.to_excel(train_file_name)
    test_data.to_excel(test_file_name)


    #########################################

# only run the following two lines if you want to generate new
# train data 
language_detector = LanguageDetectorBuilder.from_all_languages().build()
tqdm.pandas()
#rawDATAPATH = r"data\user_reviews.csv"
#prepare_train_data(rawDATAPATH)



train_datapath = r"train_data.xlsx"
df_train = read_processed_data(train_datapath)

# print("Details about Train Data")
# print(df_train.info())
train_grade_counts = df_train['grade'].value_counts()
print(train_grade_counts)

# further pre-processing of tokenized text
# list of tokens from here: https://gist.github.com/sebleier/554280
stopwords = """
i
me
my
myself
we
our
ours
ourselves
you
your
yours
yourself
yourselves
he
him
his
himself
she
her
hers
herself
it
its
itself
they
them
their
theirs
themselves
what
which
who
whom
this
that
these
those
am
is
are
was
were
be
been
being
have
has
had
having
do
does
did
doing
a
an
the
and
but
if
or
because
as
until
while
of
at
by
for
with
about
against
between
into
through
during
before
after
above
below
to
from
up
down
in
out
on
off
over
under
again
further
then
once
here
there
when
where
why
how
all
any
both
each
few
more
most
other
some
such
no
nor
not
only
own
same
so
than
too
very
s
t
can
will
just
don
should
now"""

stopwords = stopwords.split()

tokenizers = ["ws", "bpe", "rulebased", "bert"]
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
somajo_tokenizer = SoMaJo(language="en_PTB")

# load BERT vocabulary 
vocab = bert_tokenizer.get_vocab()

##############################################
##### WHITE-SPACE TOKENIZATION ###############
##############################################

# add column with tokenized tokens by whitespance
df_train['tokenized_ws'] = df_train['text'].progress_apply(tokenizer_whitespace)
df_train['tokenized_ws'] = df_train['tokenized_ws'].progress_apply(lambda x: remove_stopwords(x, stopwords))

# get vocabulary from white-space tokenization
vocab_ws = get_vocababulary(df_train, "tokenized_ws")
unknowun_vocab = unknown_tokens("ws", vocab_ws, vocab)
print(f"Unknown Vocabs: {unknowun_vocab}")

counter_ws, train_input_ids_ws, train_attention_masks_ws = encode_data(df_train, "ws", bert_tokenizer)
unknown_ws = count_unknown(train_input_ids_ws)

##############################################
############# BPE TOKENIZATION ###############
##############################################
# add column with tokenized tokens by bpe

vocab_bpe, merge_rules = learn_bpe(df_train)
unknowun_vocab = unknown_tokens("bpe",vocab_bpe, vocab)
print(f"Unknown Vocabs: {unknowun_vocab}")
df_train["tokenized_bpe"] = df_train["text"].progress_apply(lambda x: tokenizer_bpe(x, merge_rules))
df_train['tokenized_bpe'] = df_train['tokenized_bpe'].progress_apply(lambda x: remove_stopwords(x, stopwords))
vocab_bpe = get_vocababulary(df_train, "tokenized_bpe")
counter_bpe, train_input_ids_bpe, train_attention_masks_bpe = encode_data(df_train, "bpe", bert_tokenizer)
unknown_bpe = count_unknown(train_input_ids_bpe)

##############################################
############# RULE BASED ################
##############################################
# add column with tokenized tokens by rule based

# Initialize the tokenizer


# Apply the function and pass the tokenizer as an argument
df_train["tokenized_rulebased"] = df_train["text"].progress_apply(lambda x: tokenizer_rule_based(x, somajo_tokenizer))
df_train['tokenized_rulebased'] = df_train['tokenized_rulebased'].progress_apply(lambda x: remove_stopwords(x, stopwords))
vocab_rulebased = get_vocababulary(df_train, "tokenized_rulebased")
unknowun_vocab = unknown_tokens("rb", vocab_rulebased, vocab)
print(f"Unknown Vocabs: {unknowun_vocab}")
counter_rulebased, train_input_ids_rulebased, train_attention_masks_rulebased = encode_data(df_train, "rulebased", bert_tokenizer)
unknown_rulebased = count_unknown(train_input_ids_rulebased)

##############################################
################## BERT ######################
##############################################
# add column with tokenized tokens by bert

df_train["tokenized_bert"] = df_train["text"].progress_apply(lambda x: tokenizer_bert(x, bert_tokenizer))
df_train['tokenized_bert'] = df_train['tokenized_bert'].progress_apply(lambda x: remove_stopwords(x, stopwords))
vocab_bert = get_vocababulary(df_train, "tokenized_bert")
counter_bert, train_input_ids_bert, train_attention_masks_bert = encode_data(df_train, "bert", bert_tokenizer)
unknown_bert = count_unknown(train_input_ids_bert)


##############################################
############### ABOUT TOKENIZATION ###########
##############################################

with open(f"./tokenization_info.txt", "a", encoding="UTF-8") as file:
    file.write("Whistespace:\n")
    file.write(f"Vocabulary: {len(vocab_ws)}\n")
    file.write(f"Number of unknown tokens: {unknown_ws}\n")
    file.write(f"Number of times, input IDs hat >512 elements: {counter_ws}\n")
    file.write("\nBPE:\n")
    file.write(f"Vocabulary: {len(vocab_bpe)}\n")
    file.write(f"Number of unknown tokens: {unknown_bpe}\n")
    file.write(f"Number of times, input IDs hat >512 elements: {counter_bpe}\n")


    file.write("\nRule based:\n")
    file.write(f"Vocabulary: {len(vocab_rulebased)}\n")
    file.write(f"Number of unknown tokens: {unknown_rulebased}\n")
    file.write(f"Number of times, input IDs hat >512 elements: {counter_rulebased}\n")


    file.write("\nBERT:\n")
    file.write(f"Vocabulary: {len(vocab_bert)}\n")
    file.write(f"Number of unknown tokens: {unknown_bert}\n")
    file.write(f"Number of times, input IDs hat >512 elements: {counter_bert}\n")

        

##############################################
################## LABELS ####################
##############################################

train_labels = torch.tensor(df_train["grade"].tolist())

###########################################################
################## PREPARE TEST DATA ######################
###########################################################


#####################
# prepare test data #
#####################
test_datapath = r"test_data.xlsx"
df_test = read_processed_data(test_datapath)


# keep only English reviews 
df_en_test = remove_other_languages(df_test, language_detector)
# clean sentence (duplicates)
df_en_test['text'] = df_en_test['text'].progress_apply(clean_reviews_duplicates)
df_en_test['text'] = df_en_test['text'].progress_apply(clean_reviews)
test_grade_counts = df_en_test['grade'].value_counts()

df_test = copy.deepcopy(df_en_test)

for tokenization_approach in tokenizers:

    if tokenization_approach == "ws":
        df_test['tokenized_ws'] = df_test['text'].progress_apply(tokenizer_whitespace)
    elif tokenization_approach == "bpe":
        df_test['tokenized_bpe'] = df_test['text'].progress_apply(lambda x: tokenizer_bpe(x, merge_rules))
    elif tokenization_approach == "rulebased":
        df_test['tokenized_rulebased'] = df_test['text'].progress_apply(lambda x: tokenizer_rule_based(x, somajo_tokenizer))
    elif tokenization_approach == "bert":
        df_test['tokenized_bert'] = df_test['text'].progress_apply(lambda x: tokenizer_bert(x, bert_tokenizer))
    
    df_test[f'tokenized_{tokenization_approach}'] = df_test[f'tokenized_{tokenization_approach}'].progress_apply(lambda x: remove_stopwords(x, stopwords))

_, test_input_ids_ws, test_attention_masks_ws = encode_data(df_test, "ws", bert_tokenizer)
_, test_input_ids_bpe, test_attention_masks_bpe = encode_data(df_test, "bpe", bert_tokenizer)
_, test_input_ids_rulebased, test_attention_masks_rulebased = encode_data(df_test, "rulebased", bert_tokenizer)
_, test_input_ids_bert, test_attention_masks_bert = encode_data(df_test, "bert", bert_tokenizer)

test_labels = torch.tensor(df_test["grade"].tolist())
    
#############################################################
############## FINE TUNE BERT ###############################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

max_len = 0

# For every sentence...
for review in train_input_ids_ws:

    # Update the maximum sentence length.
    max_len = max(max_len, len(review))

print('Max sentence length: ', max_len)


train_data_ws = TensorDataset(train_input_ids_ws, train_attention_masks_ws, train_labels)
print(f'training samples: {train_data_ws}')

train_data_bpe = TensorDataset(train_input_ids_bpe, train_attention_masks_bpe, train_labels)
train_data_rulebased = TensorDataset(train_input_ids_rulebased, train_attention_masks_rulebased, train_labels)
train_data_bert = TensorDataset(train_input_ids_bert, train_attention_masks_bert, train_labels)

test_data_ws = TensorDataset(test_input_ids_ws, test_attention_masks_ws, test_labels)
test_data_bpe = TensorDataset(test_input_ids_bpe, test_attention_masks_bpe, test_labels)
test_data_rulebased = TensorDataset(test_input_ids_rulebased, test_attention_masks_rulebased, test_labels)
test_data_bert = TensorDataset(test_input_ids_bert, test_attention_masks_bert, test_labels)


batch_size = 8

train_size = int(0.8 * len(train_input_ids_bpe))
val_size = len(train_input_ids_bpe)  - train_size

train_data_ws, val_data_ws = random_split(train_data_ws, [train_size, val_size])
train_data_bpe, val_data_bpe = random_split(train_data_bpe, [train_size, val_size])
train_data_rulebased, val_data_rulebased = random_split(train_data_rulebased, [train_size, val_size])
train_data_bert, val_data_bert = random_split(train_data_bert, [train_size, val_size])

# #############################
# ### DATA LOADER ####
train_dataloader_ws = data_loader(train_data_ws, batch_size)
train_dataloader_bpe = data_loader(train_data_bpe, batch_size)
train_dataloader_rulebased = data_loader(train_data_rulebased, batch_size)
train_dataloader_bert = data_loader(train_data_bert, batch_size)

val_dataloader_ws = data_loader(val_data_ws, batch_size)
val_dataloader_bpe = data_loader(val_data_bpe, batch_size)
val_dataloader_rulebased = data_loader(val_data_rulebased, batch_size)
val_dataloader_bert = data_loader(val_data_bert, batch_size)

test_dataloader_ws = data_loader(test_data_ws, batch_size)
test_dataloader_bpe = data_loader(test_data_bpe, batch_size)
test_dataloader_rulebased = data_loader(test_data_rulebased, batch_size)
test_dataloader_bert = data_loader(test_data_bert, batch_size)


experiments = {

    "bertOG": {
        "model": "bert",
        "heads": 12,
        "train": train_dataloader_bert,
        "val": val_dataloader_bert,
        "test": test_dataloader_bert
    },

    "wsOG": {
        "model": "ws",
        "heads": 12,
        "train": train_dataloader_ws,
        "val": val_dataloader_ws,
        "test": test_dataloader_ws
    },

    "bpeOG": {
        "model": "bpe",
        "heads": 12,
        "train": train_dataloader_bpe,
        "val": val_dataloader_bpe,
        "test": test_dataloader_bpe
    },
    
    "rulebasedOG": {
        "model": "rulebased",
        "heads": 12,
        "train": train_dataloader_rulebased,
        "val": val_dataloader_rulebased,
        "test": test_dataloader_rulebased
    }
    
}


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def calculate_metrics(preds, labels):
    # Flatten predictions and labels
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    # Accuracy
    accuracy = np.sum(pred_flat == labels_flat) / len(labels_flat)
    
    # Pre, Recall, F1 Score
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_flat, pred_flat, average='macro'
    )

    
    return {
        'acc': accuracy,
        'prec': precision,
        'rec': recall,
        'f1': f1,
    }

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(timedelta(seconds=elapsed_rounded))



def finetune_bert(experiment_name, experiment):

    train_dataloader = experiment["train"]
    validation_dataloader = experiment["val"]

    # apparently, this does not work -.-
    # config = BertConfig.from_pretrained('bert-base-cased')
    # config.num_attention_heads = experiment["heads"]
    # config.num_labels = 6

    # model = BertForSequenceClassification.from_pretrained(
    #     "bert-base-cased",  # Pre-trained model weights
    #     config=config  # Override the configuration with your changes
    # )

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-cased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 6, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
        )


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                    lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                    eps = 1e-8, # args.adam_epsilon  - default is 1e-8.
                    weight_decay=0.01
                    )

    # Number of training epochs. The BERT authors recommend between 2 and 4. 
    # We chose to run for 4, but we'll see later that this may be over-fitting the
    # training data.
    epochs = 3

    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = int(0.1 * total_steps), # Default value in run_glue.py
                                                num_training_steps = total_steps)


    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    best_eval_accuracy = 0
    early_stop_count = 0
    early_stop_threshold = 3

    # accumulate gradients over several batches to 
    # simulate larger batch sizes
    accumulation_steps = 2
    training_stats = []
    
    with open(f"./results/protocol_{experiment_name}", "a", encoding="UTF-8") as file:
            file.write(f"\n{experiment}")

    # For each epoch...
    for epoch_i in range(0, epochs):
        
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        with open(f"./results/protocol_{experiment_name}", "a", encoding="UTF-8") as file:
            file.write('\n======== Epoch {:} / {:} ========\n'.format(epoch_i + 1, epochs))
            file.write("\nTraining...")
        
        # Measure how long the training epoch takes.
        t0 = time.time()
        total_train_loss = 0
        model.train()
        total_iterations = len(train_dataloader)
        progressBar = tqdm(total=total_iterations, desc=f"Fine-tune on {experiment_name}...")
        

        for step, batch in enumerate(train_dataloader):
            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the device using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Zero the gradients every accumulation step
            if step % accumulation_steps == 0:
                optimizer.zero_grad()

            # predict labels for input ids
            output = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels) 
                
            # compute loss of batch
            loss = output.loss / accumulation_steps

            with open(f"./results/protocol_{experiment_name}", "a", encoding="UTF-8") as file:
                file.write(f"\nLoss: {loss}")

            # add loss of batch to total loss 
            total_train_loss += loss.item()
            # Perform a backward pass to calculate the gradients
            loss.backward()

            if (step + 1) % accumulation_steps == 0:

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()  # Update the learning rate
            
            progressBar.update(1)

        progressBar.close()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)            
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        with open(f"./results/protocol_{experiment_name}", "a", encoding="UTF-8") as file:
            file.write("\n  Average training loss: {0:.2f}".format(avg_train_loss))
            file.write("\n  Training epoch took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.
        print("")
        print("Running Validation...")

        with open(f"./results/protocol_{experiment_name}", "a", encoding="UTF-8") as file:
            file.write("")
            file.write("Running Validation...")

        t0 = time.time()
        
        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()
        
        # Tracking variables 
        total_eval_loss = 0
        best_eval_loss = 2

        epoch_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
        }
        
        # Evaluate data for this epoch
        for b, batch in enumerate(validation_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        
                # predict labels 
                output= model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
                
            # Move logits and labels to CPU if we are using GPU
            logits = output.logits.detach().cpu().numpy()
            label_ids = b_labels.cpu().numpy()

            loss = output.loss
            total_eval_loss += loss.item()
            
            
            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            metrics = calculate_metrics(logits, label_ids)

            # Store metrics for this epoch
            epoch_metrics['accuracy'].append(metrics['acc'])
            epoch_metrics['precision'].append(metrics['prec'])
            epoch_metrics['recall'].append(metrics['rec'])
            epoch_metrics['f1'].append(metrics['f1'])


            with open(f"./results/protocol_{experiment_name}", "a", encoding="UTF-8") as file:
                file.write("\n=====BATCH METRICS=====")
                file.write(f"\n  Batch: {b}")
                file.write(f"\n  Accuracy: {metrics['acc']:.2f}")
                file.write(f"\n  Precision (Macro): {metrics['prec']:.2f}")
                file.write(f"\n  Recall (Macro): {metrics['rec']:.2f}")
                file.write(f"\n  F1-Score (Macro): {metrics['f1']:.2f}")
        
        # Report the final accuracy for this validation run.
        # avg_val_acc = total_eval_metrics["acc"] / len(validation_dataloader)
        avg_accuracy = np.mean(epoch_metrics['accuracy'])
        avg_precision = np.mean(epoch_metrics['precision'])
        avg_recall = np.mean(epoch_metrics['recall'])
        avg_f1 = np.mean(epoch_metrics['f1'])

        print("\n======== Training Complete ========")
        print(f"Average Accuracy over all epochs: {avg_accuracy:.2f}")
        print(f"Average Precision (Macro): {avg_precision:.2f}")
        print(f"Average Recall (Macro): {avg_recall:.2f}")
        print(f"Average F1-Score (Macro): {avg_f1:.2f}")

        with open(f"./results/protocol_{experiment_name}", "a", encoding="UTF-8") as file:
            file.write("\n=====EPOCH METRICS=====")
            file.write("\n======== Training Complete ========")
            file.write(f"\nAverage Accuracy over all epochs: {avg_accuracy:.2f}")
            file.write(f"\nAverage Precision (Macro): {avg_precision:.2f}")
            file.write(f"\nAverage Recall (Macro): {avg_recall:.2f}")
            file.write(f"\nAverage F1-Score (Macro): {avg_f1:.2f}")

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        with open(f"./results/protocol_{experiment_name}", "a", encoding="UTF-8") as file:
                file.write("\n  Validation Loss: {0:.2f}".format(avg_val_loss))
                file.write("\n  Validation took: {:}".format(validation_time))


        print(f"avg_val_loss: {avg_val_loss}")
        print(f"best_eval_accuracy: {best_eval_loss}")
        if best_eval_loss > avg_val_loss:
            best_eval_loss = avg_val_loss
            early_stop_count = 0
            print("--save model--")
            torch.save(model.state_dict(), f'trained_models/bert_model_{experiment_name}.pth')
            print("--saved new model--")
            finetuned_model = torch.load(f'trained_models/bert_model_{experiment_name}.pth')
        else:
            early_stop_count += 1

        if early_stop_count >= early_stop_threshold:
            print("Early stopping triggered!")
            with open(f"./results/protocol_{experiment_name}", "a", encoding="UTF-8") as file:
                file.write("\nEarly stopping triggered!")
            
            # progressBar.close()
            break

        
        # Record all statistics from this epoch.
        training_stats.append(
            {
                "model": f"{experiment_name}",
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time,
                "avg_accuracy": np.mean(epoch_metrics['accuracy']),
                "avg_precision": np.mean(epoch_metrics['precision']),
                "avg_recall": np.mean(epoch_metrics['recall']),
                "avg_f1": np.mean(epoch_metrics['f1']),
            }
        )

        # Save training statistics to CSV
        df_stats = pd.DataFrame(training_stats)
        df_stats.to_csv(f"training_stats_{experiment_name}.csv", index=False)


for experiment_name, experiment in experiments.items():
    # finetune_bert(experiment_name, experiment)
    
    model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=6)
    model.load_state_dict(torch.load(f'trained_models/bert_model_{experiment_name}.pth', map_location=torch.device('cpu')))

    model.eval()

    predictions = list()
    gold_labels = list()

    for batch in experiment["test"]:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        with torch.no_grad():        
            output= model(b_input_ids, 
                                   token_type_ids=None, 
                                   attention_mask=b_input_mask)
            logits = output.logits
            logits = logits.detach().cpu().numpy()
            pred_flat = np.argmax(logits, axis=1).flatten()
            label_flat = b_labels.cpu().numpy().flatten()
            
            predictions.extend(list(pred_flat))
            gold_labels.extend(list(label_flat))

    accuracy = accuracy_score(gold_labels, predictions)
    report = classification_report(gold_labels, predictions, output_dict=True)  # Get metrics for all classes
    

    results = {
        "experiment_name": experiment_name,
        "predictions": predictions,
        "gold_labels": gold_labels,
        "accuracy": accuracy,
        "classification_report": report
    }

    # Save results to a JSON file
    json_filename = f"results/{experiment_name}_test_results.json"
    with open(f"{json_filename}.txt", "a", encoding="UTF-8") as file:
            file.write(f"\n{results}")

