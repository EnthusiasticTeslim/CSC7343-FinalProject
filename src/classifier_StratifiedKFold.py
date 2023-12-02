# from src.classifier import ModelTrainer
import sys
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import StratifiedKFold

from src.model import clf_loss_func, TCRModel # model, loss function

# """
#     package versions:
#         torch: 2.1.1+cu121
#         transformers: 4.35.2
#         sklearn: 1.3.0
#         logging: 0.5.1.2
# """

def collate_fn(batch):
    return {key: torch.stack([torch.tensor(val[key]) for val in batch]) for key in batch[0]}


# key reference: 
#               https://github.com/aws-samples/amazon-sagemaker-protein-classification/blob/main/code/train.py
#               https://medium.com/analytics-vidhya/bert-pre-training-fine-tuning-eb574be614f6
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

    def __init__(self, train=True, seed = 2023, lr=2e-5, epochs=1000, log_interval=10):
        super(ModelTrainer, self).__init__()
        self.seed = seed 
        self.epochs = epochs 
        self.lr = lr    
        self.log_interval = log_interval 
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.model = TCRModel().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_func = clf_loss_func #FocalLoss(gamma=3, alpha=0.25, no_agg=True)    

    def validate(self, val_loader, model, loss_func):
        """Evaluate the network on the entire validation (part of training data) set."""

        loss_accum = []
        model.eval()
        with torch.set_grad_enabled(False):

            for data in val_loader:

                input_ids = data['input_ids'].to(self.device)
                input_mask = data['attention_mask'].to(self.device)
                labels = data['interaction'].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=input_mask)

                loss = loss_func(input=outputs, target=labels)
                loss_accum.append(loss.sum().item())


        return np.mean(loss_accum)  

    def test(self, test_loader, model, loss_func):
        """Evaluate the network on the entire test set."""
            
        model.eval()
        sum_losses = []
        correct_predictions = 0
            
        with torch.no_grad():
            for data in test_loader:
                    
                input_ids = data['input_ids'].to(self.device)
                input_mask = data['attention_mask'].to(self.device)
                labels = data['interaction'].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=input_mask)
                    
                loss = loss_func(input=outputs, target=labels)

                correct_predictions += torch.sum(torch.max(outputs, dim=1)[1] == labels)
                sum_losses.append(loss.sum().item())
                        
            print('\nTest set: loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
                    np.mean(sum_losses), 100. * correct_predictions.double() / len(test_loader.dataset)))

    def train(self, train_loader, test_loader, fold = 3, batch_size = 32):
        '''Train the network on the training set using cross validation.'''

        print(f"Training on: {self.device}")

        torch.manual_seed(self.seed) # set the seed for generating random numbers

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        
        # split data for K-fold cross validation to avoid overfitting
        indices = list(range(len(train_loader.dataset)))
        skf = StratifiedKFold(n_splits=fold, shuffle=True)

        for cv_index, (train_indices, valid_indices) in enumerate(skf.split(indices)):

            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(valid_indices)

            train_loader = DataLoader(train_loader.dataset, batch_size=batch_size,
                                                       sampler=train_sampler,
                                                       shuffle=False, collate_fn=collate_fn, pin_memory=True)
            val_loader = DataLoader(train_loader.dataset, batch_size=batch_size,
                                                     sampler=valid_sampler,
                                                     shuffle=False, collate_fn=collate_fn, pin_memory=True)
            epoch_train_loss = []
            for epoch in range(0, self.epochs + 1):

                self.model.train()
                
                for data in train_loader:

                    #print(data['input_ids'])

                    input_ids = data['input_ids'].to(self.device)   # amino acid index numbers
                    input_mask = data['attention_mask'].to(self.device) # attention mask (1 for non-padding token and 0 for padding)
                    labels = data['interaction'].to(self.device) # True for classification task

                    outputs = self.model(
                                        input_ids = input_ids, attention_mask = input_mask)
                 
                    loss = self.loss_func(input=outputs, target=labels)
                   
                    epoch_train_loss.append(loss.sum().item())

                    loss.mean().backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    

                # train & validation error after every epoch
                #print(epoch_train_loss)
                train_loss_avg = np.mean(epoch_train_loss)
                val_loss_avg = self.validate(val_loader=val_loader, model=self.model, loss_func=self.loss_func)

                print("CV: {}...At end of Epoch: {}/{}, Training Loss: {:.4f},  Validation Loss: {:.4f}".format(cv_index,
                                epoch, self.epochs, train_loss_avg, val_loss_avg))
                
        # model testing
        print('Testing the model...')
        self.test(test_loader=test_loader, model=self.model, loss_func=self.loss_func)
        print('Finished training & testing the model.')

    def save(self, path):
        """Save the model to the path specified."""

        self.model.save(path)