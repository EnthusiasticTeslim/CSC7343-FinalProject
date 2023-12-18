import torch
import torch.nn as nn
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset  # Importing Hugging Face Dataset as HFDataset to avoid confusion with PyTorch Dataset

import time, random, datasets, evaluate 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer,  TrainingArguments
from transformers import BertModel, BertConfig

from src.model import BERT_CONFIG

start_time = time.time()

## load data
data = pd.read_csv('./data/data.csv')
# Calculate the maximum length for the sequences
max_antigen_len = max([len(x) for x in data['antigen']])
max_TCR_len = max([len(x) for x in data['TCR']])
data.drop(columns=['interaction'], inplace=True)
# Apply a lambda function to insert spaces between characters
data['antigen'] = data['antigen'].apply(lambda x: ' '.join(list(x)))
data['TCR'] = data['TCR'].apply(lambda x: ' '.join(list(x)))

## Tokenizer data
config = BERT_CONFIG
tokenizer = AutoTokenizer.from_pretrained("antigen", config=config)
tokenizer.model_max_length = 64

# Put into Hugging Face dataset
dataset = HFDataset.from_pandas(data)
dataset = dataset.train_test_split(test_size=0.2)

column_names = data.columns.tolist()

# 3 for [CLS], [SEP], [PAD] and spaces between characters in the sequences
max_len = max_antigen_len + max_TCR_len + 3
print(f"max_len: {max_len}")
def tokenize_function(examples):
    return tokenizer(examples[column_names[0]], examples[column_names[1]], max_length=max_len, padding='max_length', truncation=True, return_tensors="pt")

        
tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on dataset"
        )

train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]

# evaluate function
metric = evaluate.load("accuracy")
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by prepxrocess_logits_for_metrics
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    return metric.compute(predictions=preds, references=labels)

mlm_probability = 0.15 # Percentage of data to mask

data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=mlm_probability) # data_collator for masked language modeling

# Train and evaluate
epochs = 20
model_name = f'./trainedBERT_noNSP_epochs{epochs}'
training_args = TrainingArguments(output_dir=f'{model_name}', evaluation_strategy="epoch",
                                  learning_rate=2e-5, per_device_train_batch_size=16,
                                  per_device_eval_batch_size=16, num_train_epochs=epochs,
                                  weight_decay=0.01, save_safetensors=False)
# Initialize our Trainer
model = AutoModelForMaskedLM.from_config(config)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,)

train_result = trainer.train()
trainer.save_model()  # Saves the tokenizer too for easy upload

metrics = train_result.metrics
metrics["train_samples"] = len(train_dataset)
trainer.save_metrics("train", metrics)


metrics = trainer.evaluate()
metrics["eval_samples"] = len(eval_dataset)
trainer.save_metrics("eval", metrics)

trainer.save_state()

end_time = time.time()
print(f"Time taken: {(end_time - start_time)/60} mins")
