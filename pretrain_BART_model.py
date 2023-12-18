import torch
import torch.nn as nn
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset  # Importing Hugging Face Dataset as HFDataset to avoid confusion with PyTorch Dataset

import time, random, datasets, evaluate 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer,  TrainingArguments
from typing import Optional, Tuple
from transformers import DataCollatorForLanguageModeling
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from src.model import BERT_CONFIG

## function
class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):

    def __init__(self, tokenizer: PreTrainedTokenizerBase, mlm: bool=True, mlm_probability: float=0.15, lambda_poisson: int=3):
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)
        self.lambda_poisson =lambda_poisson

    def mask_tokens(self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = inputs.clone()
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        # Masking logic
        i = 0
        while i < len(inputs):
            if special_tokens_mask[i] or np.random.rand() > self.mlm_probability:
                i += 1
                continue

            span_length = np.random.poisson(self.lambda_poisson)
            span_end = min(i + span_length, len(inputs))

            # Decide the masking strategy for the span
            masking_strategy = np.random.choice(["mask", "delete", "random", "unchanged"], p=[0.8, 0.1, 0.05, 0.05])

            if masking_strategy == "mask":
                inputs[i:span_end] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
                labels[i+1:span_end] = -100
            elif masking_strategy == "delete":
                inputs = torch.cat([inputs[:i], inputs[span_end:]])
                labels = torch.cat([labels[:i], labels[span_end:]])
                special_tokens_mask = torch.cat([special_tokens_mask[:i], special_tokens_mask[span_end:]])
                continue  # Skip the increment of i since we've shortened the sequence
            elif masking_strategy == "random":
                random_words = torch.randint(len(self.tokenizer), (span_end - i,), dtype=torch.long)
                inputs[i:span_end] = random_words
                labels[i+1:span_end] = -100
            # else: unchanged, do nothing

            i = span_end

        # Remaining tokens that are not masked are also set to -100
        labels[inputs == labels] = -100
        return inputs, labels

    def apply_deletion(self, inputs: torch.Tensor, labels: torch.Tensor, delete_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs[~delete_mask], labels[~delete_mask]

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

data_collator = CustomDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=mlm_probability)

# Train and evaluate
epochs = 50
model_name = f'./trainedBART_epochs{epochs}'
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
