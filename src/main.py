# import libraries
import torch
import torch.nn as nn

import random, datasets, evaluate
import numpy as np
import pandas as pd
import os

from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer,  TrainingArguments
from transformers import BertModel, BertConfig

from classifier import ModelTrainer
from dataPreprocessing import ProteinDataset # dataset preprocessor
from model import BERT_CONFIG

os.chdir('../')
## the data
data = pd.read_csv('./data/data.csv') # antigen, TCR, interaction
data['antigen'] = data['antigen'].apply(lambda x: ' '.join(list(x)))
data['TCR'] = data['TCR'].apply(lambda x: ' '.join(list(x)))

ANTIGEN = np.asarray(data.antigen, dtype='str')
TCR = np.asarray(data.TCR, dtype='str')
interaction = np.asarray(data.interaction, dtype="int")
print("data successfully loaded.")

## TOKENIZER
tokenizer_name = "models/antigen"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, config=BERT_CONFIG)
tokenizer.model_max_length = 64
print("Tokenizer downloaded successfully.")

tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
print ("Tokenize the first sentence:")
print (tokenized_texts[0])
# Set the maximum sequence length. The longest sequence in our training set is 47, but we'll leave room on the end anyway.
# In the original paper, the authors used a length of 512.
MAX_LEN = 128

# Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

# Pad our input tokens
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
# create attention masks
attention_masks = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask)

