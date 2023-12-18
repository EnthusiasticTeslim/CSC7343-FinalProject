import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split

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

# Example usage
# dataset = AntigenTCRDataset('../data/data.csv')
# train_dataset, test_dataset = dataset.get_train_test_datasets()

# import torch
# from torch.utils.data import Dataset
# # ref: https://medium.com/analytics-vidhya/bert-pre-training-fine-tuning-eb574be614f6

# class ProteinDataset(Dataset):
#     """
#     sequence: protein sequence containing antigen & TCR
#     interaction: 1 if antigen & TCR interact, 0 otherwise 
#     tokenizer: BERT tokenizer
#     max_len: max length of the protein sequence
#     """
#     def __init__(self, antigen, tcr, interaction, tokenizer, max_len=64):
#         self.antigen = antigen
#         self.tcr = tcr
#         self.interactions = interaction
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.antigen)

#     def __getitem__(self, item):

#         antigen = self.antigen[item]
#         tcr = self.tcr[item]
#         interaction = self.interactions[item]
#         encoding = self.tokenizer(
#                                     antigen, tcr,
#                                     return_special_tokens_mask=False,
#                                     padding='longest', truncation='longest_first', 
#                                     return_tensors="pt")
#         return {
#           'input_ids': encoding['input_ids'].flatten(),
#           'attention_mask': encoding['attention_mask'].flatten(),
#           'interaction': torch.tensor(interaction, dtype=torch.long)
#         }





# import pandas as pd
# from torch.utils.data import Dataset
# from datasets import Dataset as HFDataset  # Importing Hugging Face Dataset as HFDataset to avoid confusion with PyTorch Dataset
# from transformers import PreTrainedTokenizerBase  # Assuming you're using a tokenizer from the transformers library

# class ProteinDataset(Dataset):
#     def __init__(self, file='./data/data.csv', tokenizer = tokenizer, max_len=64):
#         self.file = file
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#         self.column_names = ['antigen', 'TCR', 'interaction']
#         self.data = self.get_data()  # Call the get_data method to load and preprocess data

#     def get_data(self):
#         # Read the csv file
#         data = pd.read_csv(self.file)

#         # Apply a lambda function to insert spaces between characters
#         data['antigen'] = data['antigen'].apply(lambda x: ' '.join(list(x)))
#         data['TCR'] = data['TCR'].apply(lambda x: ' '.join(list(x)))

#         # Put into Hugging Face dataset
#         dataset = HFDataset.from_pandas(data)
#         dataset = dataset.train_test_split(test_size=0.2)

#         def tokenize_function(examples):
#             return self.tokenizer(examples['antigen'], examples['TCR'], 
#                                   max_length=self.max_len, padding='max_length', truncation=True, return_tensors="pt")
        
#         tokenized_datasets = dataset.map(
#             tokenize_function,
#             batched=True,
#             remove_columns=self.column_names[:2],
#             desc="Running tokenizer on dataset"
#         )

#         return tokenized_datasets

#     def __getitem__(self, idx):
#         #tokenized_datasets['train'][0]['input_ids']
#         return {
#            'train': {'input_ids': self.data['train'][idx]['input_ids'],  
#             'attention_mask': self.data['train'][idx]['attention_mask'],
#             'interaction': self.data['train'][idx]['interaction'] },

#             'test': {'input_ids': self.data['test'][idx]['input_ids'], 
#             'attention_mask': self.data['test'][idx]['attention_mask'],
#             'interaction': self.data['test'][idx]['interaction'] }
#         }

#     def __len__(self):
#         return len(self.data['train'])
