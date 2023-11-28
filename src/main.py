
# import libraries
from classifier import ModelTrainer
from dataPreprocessing import ProteinDataset # dataset preprocessor


MAX_LEN = 512  # this is the max length of the sequence
PRE_TRAINED_MODEL_NAME = 'Rostlab/prot_bert_bfd_localization'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case=False)