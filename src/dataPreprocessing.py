
import torch
from torch.utils.data import Dataset
# ref: https://medium.com/analytics-vidhya/bert-pre-training-fine-tuning-eb574be614f6

class ProteinDataset(Dataset):
    """
    sequence: protein sequence containing antigen & TCR
    interaction: 1 if antigen & TCR interact, 0 otherwise 
    tokenizer: BERT tokenizer
    max_len: max length of the protein sequence
    """
    def __init__(self, antigen, tcr, interaction, tokenizer, max_len=64):
        self.antigen = antigen
        self.tcr = tcr
        self.interactions = interaction
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.antigen)

    def __getitem__(self, item):

        antigen = self.antigen[item]
        tcr = self.tcr[item]
        interaction = self.interactions[item]
        encoding = self.tokenizer(
                                    antigen, tcr,
                                    return_special_tokens_mask=False,
                                    padding='longest', truncation='longest_first', 
                                    return_tensors="pt")
        return {
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'interaction': torch.tensor(interaction, dtype=torch.long)
        }





import pandas as pd
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset  # Importing Hugging Face Dataset as HFDataset to avoid confusion with PyTorch Dataset
from transformers import PreTrainedTokenizerBase  # Assuming you're using a tokenizer from the transformers library

class ProteinDataset(Dataset):
    def __init__(self, file='./data/data.csv', tokenizer = tokenizer, max_len=64):
        self.file = file
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.column_names = ['antigen', 'TCR', 'interaction']
        self.data = self.get_data()  # Call the get_data method to load and preprocess data

    def get_data(self):
        # Read the csv file
        data = pd.read_csv(self.file)

        # Apply a lambda function to insert spaces between characters
        data['antigen'] = data['antigen'].apply(lambda x: ' '.join(list(x)))
        data['TCR'] = data['TCR'].apply(lambda x: ' '.join(list(x)))

        # Put into Hugging Face dataset
        dataset = HFDataset.from_pandas(data)
        dataset = dataset.train_test_split(test_size=0.2)

        def tokenize_function(examples):
            return self.tokenizer(examples['antigen'], examples['TCR'], 
                                  max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
        
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=self.column_names[:2],
            desc="Running tokenizer on dataset"
        )

        return tokenized_datasets

    def __getitem__(self, idx):
        #tokenized_datasets['train'][0]['input_ids']
        return {
           'train': {'input_ids': self.data['train'][idx]['input_ids'],  
            'attention_mask': self.data['train'][idx]['attention_mask'],
            'interaction': self.data['train'][idx]['interaction'] },

            'test': {'input_ids': self.data['test'][idx]['input_ids'], 
            'attention_mask': self.data['test'][idx]['attention_mask'],
            'interaction': self.data['test'][idx]['interaction'] }
        }

    def __len__(self):
        return len(self.data['train'])
