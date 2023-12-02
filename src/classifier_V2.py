# from src.classifier import ModelTrainer
import sys
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import KFold

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
#               https://medium.com/dataseries/k-fold-cross-validation-with-pytorch-and-sklearn-d094aa00105f
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

    def validate(self, val_loader, model, device, loss_func):
        """Evaluate the network on the entire validation (part of training data) set."""

        val_loss, val_accuracy = 0, 0
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


        return val_loss, val_accuracy

    def test(self, test_loader, model, loss_func, device):
        """Evaluate the network on the entire test set."""
            
        model.eval()
        
        test_loss, test_accuracy = 0, 0
            
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
                
        return test_loss, test_accuracy                    
    

    def train(self, model, train_loader, loss_func, optimizer, device):
        """Train the network on the training set."""
        train_loss, train_accuracy = 0, 0
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

        return train_loss, train_accuracy

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
                                                       shuffle=False, collate_fn=collate_fn, pin_memory=True)
            val_loader = DataLoader(train_loader.dataset, batch_size=batch_size,
                                                     sampler=valid_sampler,
                                                     shuffle=False, collate_fn=collate_fn, pin_memory=True)
            
            print("CV: {}".format(cv_index))

            self.history = {'train_loss': [], 'val_loss': [],'train_acc':[],'val_acc':[]}

            for epoch in range(0, self.epochs + 1):
                # Training
                epoch_train_loss, epoch_train_accuracy = self.train(model=self.model, 
                                                                    train_loader=train_loader, loss_func=self.loss_func, 
                                                                    optimizer=self.optimizer, device=self.device)
                # Validation
                epoch_val_loss, epoch_val_accuracy = self.validate(val_loader=val_loader, 
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

                # train & validation error after every epoch
                print("Epoch: {}/{}, Training Loss: {:.4f}, Training Accuracy: {:.2f} %, Validation Loss: {:.4f}, Validation Accuracy: {:.2f} %".format(
                                epoch, self.epochs, train_loss, train_accuracy, val_loss, val_accuracy))
        
        # model testing
        print('Testing the model...')
        test_loss, test_accuracy = self.test(test_loader=test_loader, model=self.model, loss_func=self.loss_func, device=self.device)
        test_loss_, test_accuracy_ = test_loss / len(test_loader.sampler), test_accuracy * 100 / len(test_loader.sampler)

        print("Test Loss: {:.4f}, Test Accuracy: {:.2f} %".format(test_loss_, test_accuracy_))
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

        print('Performance of {} fold cross validation'.format(self.fold))
        print("Average Training Loss: {:.4f} \t Average Val Loss: {:.4f} \t Average Training Acc: {:.3f} \t Average Val Acc: {:.3f}".format(avg_train_loss, avg_val_loss,avg_train_acc,avg_val_acc))  
        
        np.save(f'{path}_history.npy', self.history)
