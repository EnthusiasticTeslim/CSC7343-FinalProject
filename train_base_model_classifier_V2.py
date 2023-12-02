import torch
import torch.nn as nn
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset  # Importing Hugging Face Dataset as HFDataset to avoid confusion with PyTorch Dataset

import time, random, datasets, evaluate 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformers import AutoTokenizer

from src.model import BERT_CONFIG
from src.classifier_V2 import ModelTrainer

start_time = time.time()

## load data
data = pd.read_csv('./data/data_balanced.csv')
# Apply a lambda function to insert spaces between characters
data['antigen'] = data['antigen'].apply(lambda x: ' '.join(list(x)))
data['TCR'] = data['TCR'].apply(lambda x: ' '.join(list(x)))

## Tokenizer data
config = BERT_CONFIG
tokenizer = AutoTokenizer.from_pretrained("src/antigen", config=config)
tokenizer.model_max_length = 64

# Put into Hugging Face dataset
dataset = HFDataset.from_pandas(data)
dataset = dataset.train_test_split(test_size=0.2)

column_names = data.columns.tolist()

# def tokenize_function(examples):
#     return tokenizer(examples[column_names[0]], examples[column_names[1]], return_special_tokens_mask=False,
#                      padding='longest', truncation='longest_first', return_tensors="pt")
def tokenize_function(examples):
    return tokenizer(examples[column_names[0]], examples[column_names[1]], return_special_tokens_mask=False,
                     padding='max_length', truncation=True, return_tensors="pt")
        
tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names[:2],
            desc="Running tokenizer on dataset"
        )

# Train and evaluate
def collate_fn(batch):
    return {key: torch.stack([torch.tensor(val[key]) for val in batch]) for key in batch[0]}

batch_size = 128
train_loader = torch.utils.data.DataLoader(tokenized_datasets['train'], batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True)
test_loader = torch.utils.data.DataLoader(tokenized_datasets['test'], batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True)
# model
epoch=10
Model = ModelTrainer(epochs=epoch, lr=1e-3)
Model.execute_run(train_loader=train_loader, test_loader=test_loader, batch_size=batch_size, fold=3)

Model.save(f'./model/base_model_sum_balanced_epoch{epoch}_classified_V2')

end_time = time.time()

print(f"Total time taken: {round((end_time - start_time)/60, 2)} mins")