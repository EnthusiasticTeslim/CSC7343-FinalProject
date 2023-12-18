#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 4 15:51:00 2023
@author: Teslim Olayiwola

# Important information:

# CASE 1: ************************ How to train the base model ************************
# if copying to a new file make sure to import the following:
from project import TrainbaseModel
# create an instance of the model
baseModel = Train_Base_PretrainedModel(data_path='data.csv', split=0.2, seed=48, batch_size = 128)
# train + save
baseModel.train_and_save_model(epochs=10, fold=3, learning_rate=1e-3, model_save_path='classifier_wt_base_model')
# calculate time
baseModel.calculate_total_time()

# CASE 2: ************************ How to pretrain a the model ************************
#To pretrain the model use the following command:
pretrainer = PreTrainer(data_path, epochs=10, mlm_probability=0.15)
# train + save
pretrainer.train(method = 'BART') # method = 'BERT' or 'BART', default is 'BART', you can change it to 'BERT' if you want to pretrain a BERT model
# get pytorch version
pretrainer.get_pytorch_version(new_model_name ='pytorch_BART') # if new_model_name is not specified, the default name is pytorch_{method}
# *Important: Before running the above command, make sure to download the encoded tokenizer from the following link. 
# Also, note that the tokenizer works in the same as the python class AntigenTCRDataset
https://drive.google.com/drive/folders/18D1J2SE2p51ayuoyAnUe6UCGcI9U6vL0?usp=sharing


# CASE 3: ************************ How to train the classifier with pretrained model ************************
# if copying to a new file make sure to import the following:
from project import TrainbaseModel
# create an instance of the model
classifierModel = Train_Base_PretrainedModel(data_path='data.csv', split=0.2, seed=48, batch_size = 128)
# train + save
classifierModel.train_and_save_model(pretrained=True, epochs=10, fold=3, learning_rate=1e-3, 
                path_to_pretrained_model = '../trainedBART_epochs50/reformed_pytorch_model_BART', model_save_path='classifier_wt_pretrained_base_model')
# calculate time
classifierModel.calculate_total_time()

# CASE 4: ************************ How to use finetuned the model obtained from pretraining scheme ************************
# Link to all models are here at: https://drive.google.com/drive/folders/16AK1Qi4MWQXwbQURs_p9VcKghq7NUmK1?usp=sharing
# For base model (base_model_sum_balanced_epoch10_classified_V3.pt): 
#               https://drive.google.com/file/d/1NiPSs5KJ9t1hqSBHrLcg40Z3y6YsNYvo/view?usp=sharing
# For finetuned model trained based on approach 2 - BART (BARTpretrained_model_sum_balanced_epoch10_seed48_classified_V3.pt)
#               https://drive.google.com/file/d/1YQVCfpjWPEf-JAcjoqCCteFAY1Fu0Xtn/view?usp=sharing
# For finetuned model trained based on approach 1 - BERT (BERTpretrained_model_sum_balanced_epoch10_seed48_classified_V3.pt)
#               https://drive.google.com/file/d/196D1L250uwYsiZscadFRUA3nFCjmW4l5/view?usp=sharing
# To test the model use the following command:
# if copying to a new file make sure to import the following:
from project import ModelTrainer
# create an instance of the model
Model = ModelTrainer(load_trained_classifier=True, path_to_trained_classifier='BARTpretrained_model_sum_balanced_epoch10_seed48_classified_V3')
# apply the model on the test data
Model.predict(csv_path = './data/data_balanced.csv', batch_size = 128)


# Special thanks to the following key references: 
#               https://github.com/aws-samples/amazon-sagemaker-protein-classification/blob/main/code/train.py
#               https://medium.com/analytics-vidhya/bert-pre-training-fine-tuning-eb574be614f6
#               https://medium.com/dataseries/k-fold-cross-validation-with-pytorch-and-sklearn-d094aa00105f

# package versions:
# 
#     package versions:
#         torch: 2.1.1+cu121
#         transformers: 4.35.2
#         sklearn: 1.3.0
#         logging: 0.5.1.2
#
"""

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split

import sys, time, random, datasets, evaluate
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from model import clf_loss_func, TCRModel, BERT_CONFIG # model, loss function, and model configuration

# ********************************* Tokenizer *********************************
class AntigenTCRDataset(Dataset):
    def __init__(self, csv_file, CLS='[CLS]', SEP='[SEP]', PAD='[PAD]', UNK='[UNK]'):
        # Read the CSV file
        self.data = pd.read_csv(csv_file)

        # The total character length including [CLS], [SEP], and [PAD]
        self.total_max_length = 31 + 3  # 34 characters

        # Special tokens
        self.CLS = CLS
        self.SEP = SEP
        self.PAD = PAD
        self.UNK = UNK

        # Encoding dictionary for amino acids and special tokens
        self.encodings = {f"{self.PAD}": 0, f"{self.CLS}": 1, f"{self.SEP}": 2, "[MASK]": 3,
                            f"{self.UNK}": 4, "G": 5, "T": 6, "C": 7, "Q": 8, "N": 9, "W": 10, 
                            "E": 11, "I": 12, "Y": 13, "A": 14, "R": 15, "L": 16, "S": 17, "M": 18, "D": 19, "F": 20, "H": 21, "K": 22, "V": 23, "P": 24}

        # Process and pad the sequences
        self.data['combined_seqs'] = self.data.apply(
            lambda row: self.pad_combined_sequences(row['antigen'], row['TCR']), axis=1)

        # Convert to input tokens and attention masks
        self.input_tokens = [self.sequence_to_input_tokens(seq, self.encodings) for seq in self.data['combined_seqs']]
        self.attention_masks = [self.create_attention_mask(tokens, self.encodings) for tokens in self.input_tokens]

    def split_data(self, test_size=0.2, state=48):
        train_idx, test_idx = train_test_split(range(len(self.data)), test_size=test_size, random_state=state)
        train_dataset = Subset(self, train_idx)
        test_dataset = Subset(self, test_idx)
        return train_dataset, test_dataset
    
    def __len__(self):
        return len(self.data)

    def separate_aa(self, sequence):
        return ' '.join(sequence), len(sequence)

    def pad_combined_sequences(self, antigen_sequence, tcr_sequence):
        # Separate amino acids in each sequence
        separated_antigen, len_antigen = self.separate_aa(antigen_sequence)
        separated_tcr, len_tcr = self.separate_aa(tcr_sequence)

        # Combine sequences with special tokens
        combined = f'{self.CLS} {separated_antigen} {self.SEP} {separated_tcr} {self.SEP}'

        # Add padding to achieve the desired total length
        padding_needed = self.total_max_length - (3 + len_tcr + len_antigen)
        for _ in range(padding_needed):
            combined += ' ' + self.PAD

        return combined

    def sequence_to_input_tokens(self, sequence, encodings):
        return [encodings.get(elem, encodings[self.UNK]) for elem in sequence.split()]

    def create_attention_mask(self, input_tokens, encodings):
        return [1 if token != encodings[self.PAD] else 0 for token in input_tokens]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            'antigen': self.data.iloc[idx]['antigen'],
            'tcr': self.data.iloc[idx]['TCR'],
            'combined_sequences': self.data.iloc[idx]['combined_seqs'],
            'input_ids': torch.tensor(self.input_tokens[idx]),
            'attention_mask': torch.tensor(self.attention_masks[idx]),
            'interaction': torch.tensor(self.data.iloc[idx]['interaction'])
        }

        return sample

#********************************* Model Trainer *********************************
class ModelTrainer(nn.Module):

    """
        ************** Train/Test the model using cross validation ************** 
        seed: seed for random number generator
        epochs: number of epochs to train
        lr: learning rate
        train: flag whether to train the model
        log_interval: how many batches to wait before logging training status
        model: takes input_ids: str, attention_mask: str, classification: bool
        
    """

    def __init__(self, load_trained_classifier=False, pretrained=False, path_to_pretrained_model=None, path_to_trained_classifier=None, seed = 2023, lr=2e-5, epochs=1000):
        super(ModelTrainer, self).__init__()
        self.seed = seed 
        self.epochs = epochs 
        self.lr = lr   
        self.pretrained = pretrained  
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.model = TCRModel().to(self.device)
        if load_trained_classifier is not True:
            if self.pretrained and path_to_pretrained_model is not None:
                print(f'Loading pretrained model...{path_to_pretrained_model}.bin')
                self.model.load(f"{path_to_pretrained_model}.bin")
                print('Training a classifier with the pretrained model...')
            else:
                print('Returning to training a classifier with the base model...')
         
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            
        else:
            if path_to_trained_classifier is not None:
                print('Loading trained classifier model...')
                self.model.load(f"{path_to_trained_classifier}.pt")
                self.model = self.model.to(self.device)
            else:
                raise ValueError('Please specify the path to the trained classifier model.')

        self.loss_func = clf_loss_func #FocalLoss(gamma=3, alpha=0.25, no_agg=True)    

    def validate(self, val_loader, model, device, loss_func):
        """Evaluate the network on the entire validation (part of training data) set."""

        val_loss, val_accuracy = 0, 0
        all_labels, all_predictions = [], []

        model.eval()
        with torch.no_grad():

            for data in val_loader:
                # get the inputs
                input_ids = data['input_ids'].to(device)
                input_mask = data['attention_mask'].to(device)
                labels = data['interaction'].to(device)
                # forward pass
                outputs = model(input_ids=input_ids, attention_mask=input_mask)
                # loss and accuracy
                loss = loss_func(input=outputs, target=labels)
                val_loss += loss.sum().item() * input_ids.size(0)
                scores, predictions = torch.max(outputs, dim=1)
                val_accuracy += (predictions == labels).sum().item()

                # Store predictions and labels for AUC calculation
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(outputs.cpu().detach().numpy()[:, 1])
            
        # Compute AUC
        auc = roc_auc_score(all_labels, all_predictions)


        return val_loss, val_accuracy, auc

    def test(self, test_loader, model, loss_func, device):
        """Evaluate the network on the entire test set and calculate AUC."""

        model.eval()

        test_loss, test_accuracy = 0, 0
        all_labels, all_predictions = [], []

        with torch.no_grad():
            for data in test_loader:
                # get the inputs
                input_ids = data['input_ids'].to(device)
                input_mask = data['attention_mask'].to(device)
                labels = data['interaction'].to(device)

                # forward pass
                outputs = model(input_ids=input_ids, attention_mask=input_mask)

                # loss and accuracy
                loss = loss_func(input=outputs, target=labels)
                test_loss += loss.sum().item() * input_ids.size(0)

                scores, predictions = torch.max(outputs, dim=1)
                test_accuracy += (predictions == labels).sum().item()

                # Store predictions and labels for AUC calculation
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(outputs.cpu().detach().numpy()[:, 1])  # Assuming binary classification (1 is the positive class)

        # Compute AUC
        auc = roc_auc_score(all_labels, all_predictions)

        return test_loss, test_accuracy, auc                  
    

    def train(self, model, train_loader, loss_func, optimizer, device):
        """Train the network on the training set."""
        train_loss, train_accuracy = 0, 0
        all_labels, all_predictions = [], []

        model.train()
        
        for data in train_loader:
            
            # get the inputs
            input_ids = data['input_ids'].to(device)   # amino acid index numbers
            input_mask = data['attention_mask'].to(device) # attention mask (1 for non-padding token and 0 for padding)
            labels = data['interaction'].to(device) # True for classification task
            # forward pass
            outputs = self.model(input_ids = input_ids, attention_mask = input_mask)
            # loss and backward pass
            loss = self.loss_func(input=outputs, target=labels)
            loss.mean().backward()
            optimizer.step()
            optimizer.zero_grad()
            # loss and accuracy
            train_loss += loss.sum().item() * input_ids.size(0)
            scores, predictions = torch.max(outputs, dim=1)
            train_accuracy += (predictions == labels).sum().item()

            # auc score
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(outputs.cpu().detach().numpy()[:, 1])

        # Compute AUC
        auc = roc_auc_score(all_labels, all_predictions)

        return train_loss, train_accuracy, auc

    def execute_run(self, train_loader, test_loader, fold = 3, batch_size = 32):
        '''Train, Test and Validate the network on the training set using cross validation.'''

        print(f"Training on: {self.device}")

        torch.manual_seed(self.seed) # set the seed for generating random numbers

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        
        # split data for K-fold cross validation to avoid overfitting
        self.fold = fold
        indices = list(range(len(train_loader.dataset)))
        kf = KFold(n_splits=self.fold, shuffle=True)

        for cv_index, (train_indices, valid_indices) in enumerate(kf.split(indices)):

            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(valid_indices)

            train_loader = DataLoader(train_loader.dataset, batch_size=batch_size,
                                                       sampler=train_sampler,
                                                       shuffle=False, pin_memory=True)
            val_loader = DataLoader(train_loader.dataset, batch_size=batch_size,
                                                     sampler=valid_sampler,
                                                     shuffle=False, pin_memory=True)
            
            print("CV: {}".format(cv_index))

            self.history = {'train_loss': [], 'val_loss': [],'train_acc':[],'val_acc':[], 'train_auc':[], 'val_auc':[]}

            for epoch in range(0, self.epochs + 1):
                # Training
                epoch_train_loss, epoch_train_accuracy, auc_train = self.train(model=self.model, 
                                                                    train_loader=train_loader, loss_func=self.loss_func, 
                                                                    optimizer=self.optimizer, device=self.device)
                # Validation
                epoch_val_loss, epoch_val_accuracy, auc_val = self.validate(val_loader=val_loader, 
                                                                    model=self.model, loss_func=self.loss_func, 
                                                                    device=self.device)
                # 
                train_loss = epoch_train_loss / len(train_loader.sampler)
                train_accuracy = epoch_train_accuracy * 100 / len(train_loader.sampler)
                val_loss = epoch_val_loss / len(val_loader.sampler)
                val_accuracy = epoch_val_accuracy * 100/ len(val_loader.sampler)

                self.history['train_loss'].append(train_loss)    
                self.history['train_acc'].append(train_accuracy)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_accuracy)
                self.history['train_auc'].append(auc_train)
                self.history['val_auc'].append(auc_val)

                # train & validation error after every epoch
                print("Epoch: {}/{}, Training Loss: {:.4f}, Training Accuracy: {:.2f} %, Train AUC score: {:.2f}, Validation Loss: {:.4f}, Validation Accuracy: {:.2f} %, Validation AUC score: {:.2f}".format(
                                epoch, self.epochs, train_loss, train_accuracy, auc_train, val_loss, val_accuracy, auc_val))
            # after cross validation
            print(f'Finished training & validation the model for CV..... {cv_index} .........')
            avg_train_loss_after_CV = np.mean(self.history['train_loss'])
            avg_val_loss_after_CV = np.mean(self.history['val_loss'])
            avg_train_acc_after_CV = np.mean(self.history['train_acc'])
            avg_val_acc_after_CV = np.mean(self.history['val_acc'])
            avg_train_auc_after_CV = np.mean(self.history['train_auc'])
            avg_val_auc_after_CV = np.mean(self.history['val_auc'])
            
            print("Average Training Loss: {:.4f} \t Average Val Loss: {:.4f} \t Average Training Acc: {:.3f} \t Average Val Acc: {:.3f} \t Average Training AUC: {:.3f} \t Average Val AUC: {:.3f}".format(avg_train_loss_after_CV, avg_val_loss_after_CV,avg_train_acc_after_CV,avg_val_acc_after_CV, avg_train_auc_after_CV, avg_val_auc_after_CV))  
            
        # model testing
        print('Testing the model...')
        test_loss, test_accuracy, auc_test = self.test(test_loader=test_loader, model=self.model, loss_func=self.loss_func, device=self.device)
        test_loss_, test_accuracy_ = test_loss / len(test_loader.sampler), test_accuracy * 100 / len(test_loader.sampler)

        print("Test Loss: {:.4f}, Test Accuracy: {:.2f} %, Test AUC score: {:.2f}".format(test_loss_, test_accuracy_, auc_test))
        print('Finished training & testing the model.')

    def save(self, path):
        """Save the model to the path specified."""
        # save model
        self.model.save(f"{path}.pt")
        # save history
        avg_train_loss = np.mean(self.history['train_loss'])
        avg_val_loss = np.mean(self.history['val_loss'])
        avg_train_acc = np.mean(self.history['train_acc'])
        avg_val_acc = np.mean(self.history['val_acc'])
        avg_train_auc = np.mean(self.history['train_auc'])
        avg_val_auc = np.mean(self.history['val_auc'])

        print('Performance of {} fold cross validation'.format(self.fold))
        print("Average Training Loss: {:.4f} \t Average Val Loss: {:.4f} \t Average Training Acc: {:.3f} \t Average Val Acc: {:.3f} \t Average Training AUC: {:.3f} \t Average Val AUC: {:.3f}".format(avg_train_loss, avg_val_loss,avg_train_acc,avg_val_acc, avg_train_auc, avg_val_auc))  
        
        np.save(f'{path}_history.npy', self.history)


    def predict(self, csv_path, batch_size = 128):
        """Predict the class of the data."""
        dataset = AntigenTCRDataset(csv_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        
        all_labels, all_predictions = [], []
        pred_loss, pred_accuracy = 0, 0

        self.model.eval()
        with torch.no_grad():
            for data in dataloader:
                # get the inputs
                input_ids = data['input_ids'].to(self.device)
                input_mask = data['attention_mask'].to(self.device)
                labels = data['interaction'].to(self.device)
                # forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=input_mask)
                # loss and accuracy
                loss = self.loss_func(input=outputs, target=labels)
                pred_loss += loss.sum().item() * input_ids.size(0)
                scores, predictions = torch.max(outputs, dim=1)
                pred_accuracy += (predictions == labels).sum().item()
                # Store predictions and labels for AUC calculation
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(outputs.cpu().detach().numpy()[:, 1])
        
        # Compute AUC
        auc = roc_auc_score(all_labels, all_predictions)

        print(f'Loss: {pred_loss / len(dataloader.sampler):.2f}')
        print(f'AUC score: {auc:.2f}')
        print(f'Accuracy: {100 * pred_accuracy / len(dataloader.sampler):.2f} %')
        

        return all_predictions

# ********************************* Train the classifier with the base model or pretrained model *********************************

class Train_Base_PretrainedModel:
    '''Train the classifier with the base model or pretrained model'''
    def __init__(self, data_path, split=0.2, seed=48, batch_size = 128):
        self.start_time = time.time()
        self.batch_size = batch_size
        self.seed = seed
        dataset = AntigenTCRDataset(data_path)
        print(f'Number of data points: {len(dataset)}')
        train_dataset, test_dataset = dataset.split_data(test_size=split, state=self.seed)
        print(f'Number of training data points: {len(train_dataset)}')
        print(f'Number of testing data points: {len(test_dataset)}')
        
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)

    
    # ModelTrainer(epochs=epoch, pretrained=True, lr=1e-3, seed=seed, path_to_trained_model='./trainedBART_epochs50/reformed_pytorch_model_BART.bin')
    #  load_trained_classifier=False, pretrained=False, path_to_pretrained_model=None, path_to_trained_classifier=None, seed = 2023, lr=2e-5, epochs=1000):
    def train_and_save_model(self, pretrained=False,  epochs=10, fold=3, learning_rate=2e-3, path_to_pretrained_model = None, model_save_path='base_model'):
        '''Train and save the model
        if load_trained_classifier is False:
            set pretrained to True and the path_to_pretrained_model if you want to train a classifier with a pretrained model
            set pretrained to False and path_to_trained_model = None if you want to train a classifier with a base model'''
        self.Model = ModelTrainer(epochs=epochs, pretrained= pretrained, lr=learning_rate, path_to_pretrained_model=path_to_pretrained_model, seed=self.seed)
        self.Model.execute_run(train_loader=self.train_loader, test_loader=self.test_loader, batch_size=self.batch_size, fold=fold)
        self.Model.save(model_save_path)

    def calculate_total_time(self):
        self.end_time = time.time()
        total_time = round((self.end_time - self.start_time) / 60, 2)
        return f"Total time taken: {total_time} mins"


# ********************************* Pre train a model based on approach 1  (named BERT) or 2 (named BART)*********************************

from datasets import Dataset as HFDataset 
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer,  TrainingArguments
from typing import Optional, Tuple

class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):

    """Custom data collator for language modeling. Implement the logic as discussed in the report"""

    def __init__(self, tokenizer, mlm: bool=True, mlm_probability: float=0.15, lambda_poisson: int=3):
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

class PreTrainer:

    '''Pretrain a model based on approach 1 (BERT) or 2 (BART)'''

    def __init__(self, data_path, epochs=50, mlm_probability=0.15):
        self.data_path = data_path
        self.epochs = epochs
        self.mlm_probability = mlm_probability

    def train(self, method = 'BART'): # method = 'BERT' or 'BART', default is 'BART', you can change it to 'BERT' if you want to pretrain a BERT model

        self.method = method

        start_time = time.time()

        ## load data
        data = pd.read_csv(self.data_path)
        # Calculate the maximum length for the sequences
        max_antigen_len = max([len(x) for x in data['antigen']])
        max_TCR_len = max([len(x) for x in data['TCR']])
        data.drop(columns=['interaction'], inplace=True)
        # Apply a lambda function to insert spaces between characters
        data['antigen'] = data['antigen'].apply(lambda x: ' '.join(list(x)))
        data['TCR'] = data['TCR'].apply(lambda x: ' '.join(list(x)))

        ## Tokenizer data
        config = BERT_CONFIG  # You need to define BERT_CONFIG
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
            # by preprocess_logits_for_metrics
            labels = labels.reshape(-1)
            preds = preds.reshape(-1)
            mask = labels != -100
            labels = labels[mask]
            preds = preds[mask]
            return metric.compute(predictions=preds, references=labels)
        if self.method == 'BART':
            data_collator = CustomDataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm_probability=self.mlm_probability)
        else:
            if self.method == 'BERT':
                data_collator = DataCollatorForLanguageModeling(
                                tokenizer=tokenizer,
                                mlm_probability=mlm_probability)

        # Train and evaluate
        
        training_args = TrainingArguments(output_dir=f'{self.method}', evaluation_strategy="epoch",
                                          learning_rate=2e-5, per_device_train_batch_size=16,
                                          per_device_eval_batch_size=16, num_train_epochs=self.epochs,
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

    def get_pytorch_version(self, new_model_name ='pytorchBART' ):
        '''Get the pytorch version of the pretrained model'''
        
        if new_model_name is None:
            new_model_name = f"pytorch{self.method}"
        # based model
        model = TCRModel()
        # get the state dict from the trained model
        state_dict = torch.load(f"{self.method}/pytorch_model.bin")
        # remove the bert appendix
        new_state_dict = {}
        for key in state_dict.keys():
            new_key = key.replace("bert.", "", 1)  # the "1" ensures only the first occurrence is replaced
            new_state_dict[new_key] = state_dict[key]

        # Remove keys that start with "cls"
        keys_to_delete = [key for key in new_state_dict.keys() if key.startswith("cls")]
        for key in keys_to_delete:
            del new_state_dict[key]
        # add the new state dict to the model
        model_state_dict = model.state_dict()
        model_state_dict.update(new_state_dict)
        model.load_state_dict(model_state_dict)
        # save the model
        torch.save(model.state_dict(), f"{self.method}/{new_model_name}.bin")

        print(f"Saved model to {self.method}/{new_model_name}.bin")