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
from src.classifier_V3 import ModelTrainer
from src.dataPreprocessing import AntigenTCRDataset

start_time = time.time()

## load and tokenizer data
dataset = AntigenTCRDataset('./data/data_balanced.csv')
print(f'Number of data points: {len(dataset)}')
seed = 48
split = 0.2
train_dataset, test_dataset = dataset.split_data(test_size=split, state=seed)
print(f'Number of training data points: {len(train_dataset)}')
print(f'Number of testing data points: {len(test_dataset)}')


batch_size = 128
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
# model
epoch=10
Model = ModelTrainer(epochs=epoch, lr=1e-3, seed=seed)
Model.execute_run(train_loader=train_loader, test_loader=test_loader, batch_size=batch_size, fold=3)

Model.save(f'./model/base_model_sum_balanced_epoch{epoch}_classified_V3') #base_model_sum_balanced_epoch

end_time = time.time()

print(f"Total time taken: {round((end_time - start_time)/60, 2)} mins")