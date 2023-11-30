# import libraries
import numpy as np
import pandas as pd
import torch
from classifier import ModelTrainer
from dataPreprocessing import ProteinDataset # dataset preprocessor

from transformers import BertTokenizer

## the data
data = pd.read_csv('../data/data.csv') # antigen, TCR, interaction
ANTIGEN = np.asarray(data.antigen, dtype='str')
TCR = np.asarray(data.TCR, dtype='str')
interaction = np.asarray(data.interaction, dtype="int")

## tokenizer
MAX_LEN = 512  # this is the max length of the sequence
BASE_MODEL_NAME = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(BASE_MODEL_NAME, do_lower_case=False)

## create tokens
# Adding CLS (<cls>) and SEP(<sep>) to return "<cls>{antigen}<sep>{tcr}"
sequence = ["<cls> " + ant + " <sep> " + tcr for antigen, tcr in zip(ANTIGEN, TCR)]


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
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

