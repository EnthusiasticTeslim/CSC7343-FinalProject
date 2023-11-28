
import torch
from torch.utils.data import Dataset

class ProteinDataset(Dataset):
    """
    sequence: protein sequence containing antigen & TCR
    interaction: 1 if antigen & TCR interact, 0 otherwise 
    tokenizer: BERT tokenizer
    max_len: max length of the protein sequence
    """
    def __init__(self, sequence, interaction, tokenizer, max_len):
        self.sequence = sequence
        self.interactions = interaction
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, item):
        sequence = str(self.sequence[item])
        interaction = self.interactions[item]
        encoding = self.tokenizer.encode_plus(
                                                sequence,
                                                truncation=True,
                                                add_special_tokens=True,
                                                max_length=self.max_len,
                                                return_token_type_ids=False,
                                                padding='max_length',
                                                return_attention_mask=True,
                                                return_tensors='pt',
                                            )
        return {
          'protein': sequence,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'interaction': torch.tensor(interaction, dtype=torch.long)
        }
