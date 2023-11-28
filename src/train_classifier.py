import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from .model import BERT_CONFIG, FocalLoss, TCRModel

class ProteinDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len


class ModelTrainer(nn.Module):

    """
     package versions:
     torch: 2.1.1+cu121
     transformers: 4.35.2
     sklearn: 1.3.0
     logging: 0.5.1.2"""

    def __init__(self, args, train=True, seed = 2023, lr=2e-5, epochs=1000, log_interval=10):
        
        self.seed = seed # seed for random number generator
        self.epochs = epochs # number of epochs to train
        self.log_interval = log_interval # how many batches to wait before logging training status
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.model = TCRModel.to(self.device) # takes input_ids, attention_mask, classification
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

    def train(self, train_loader, fold = 5, batch_size = 32):
        '''Train the network on the training set using cross validation.'''
        
        # set the seed for generating random numbers
        torch.manual_seed(args.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        
         # split data for K-fold cross validation to avoid overfitting
        indices = list(range(len(train_loader.dataset)))
        kf = KFold(n_splits=fold, shuffle=True)
        cv_index = 0
        index_list_train = []
        index_list_valid = []

        for train_indices, valid_indices in kf.split(indices):

            index_list_train.append(train_indices)
            index_list_valid.append(valid_indices)

            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(valid_indices)

            train_loader = DataLoader(x.dataset, batch_size=batch_size,
                                                       sampler=train_sampler,
                                                       shuffle=False)
            val_loader = DataLoader(x.dataset, batch_size=batch_size,
                                                     sampler=valid_sampler,
                                                     shuffle=False)
            epoch_train_loss = []
            for epoch in range(0, self.epochs + 1):

                self.model.train()
                
                for step, batch in enumerate(train_loader):

                    input_ids = batch['input_ids'].to(device)
                    input_mask = batch['attention_mask'].to(device)
                    labels = batch['targets'].to(device)

                    outputs = self.model(
                                        input_ids = input_ids  # amino acid index numbers
                                        attention_mask = input_mask, # attention mask (1 for non-padding token and 0 for padding)
                                        classification = True # True for classification task
                                        )
                    self.model.to(self.device)
                    loss = clf_loss_func(input=outputs, target=labels)
                    epoch_train_loss.append(loss)

                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    if step % self.log_interval == 0:
                        logger.info("CV index: {} Train Epoch: {} [{}/{} ({:.0f}%)] Training Loss: {:.6f}".format(
                                cv_index, epoch, step * len(batch['input_ids'])*world_size,
                                len(train_loader.dataset), 100.0 * step / len(train_loader), loss))

                # error after every epoch
                train_loss_avg = np.mean(epoch_train_loss)
                val_loss_avg = self.validate(val_loader=val_loader, model=self.model, loss_func=clf_loss_func)

                logger.info("At end of Epoch: {}/{}, Training Loss: {:.4f},  Validation Loss: {:.4f}".format(
                                epoch, self.epochs, train_loss_avg, val_loss_avg))
        
        def validate(self, val_loader, model, loss_func):
            """Evaluate the network on the entire validation (part of training data) set."""

            loss_accum = []
            model.eval()
            with torch.set_grad_enabled(False):

                for (feature, label) in val_loader:
                    label = convert_labels(label) # convert labels to one-hot encoding
                    feature, label = feature.to(self.device).long(), label.to(self.device)

                    outputs = model(feature)

                    loss = loss_func(input=outputs, target=labels)
                    loss_accum.append(loss)


            return np.mean(loss_accum)  

        def test(self, test_loader):
            """Evaluate the network on the entire test set."""
            
            self.model.eval()
            sum_losses = []
            correct_predictions = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    
                    input_ids = batch['input_ids'].to(self.device)
                    input_mask = batch['attention_mask'].to(self.device)
                    labels = batch['targets'].to(self.device)

                    outputs = self.model(input=input_ids, attention_mask=input_mask, classification=True)
                    
                    loss = clf_loss_func(input=outputs, target=labels)

                    correct_predictions += torch.sum(torch.max(outputs, dim=1) == labels)
                    sum_losses.append(loss)
                        
            logger.info('\nTest set: loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
                    np.mean(sum_losses), 100. * correct_predictions.double() / len(test_loader.dataset)))
