# ********** How to use the project **********
# command: python test_project.py
# comment case by case to test the project


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
# https://drive.google.com/drive/folders/18D1J2SE2p51ayuoyAnUe6UCGcI9U6vL0?usp=sharing



# CASE 3: ************************ How to train the classifier with pretrained model ************************
# if copying to a new file make sure to import the following:
from project import TrainbaseModel
# create an instance of the model
classifierModel = Train_Base_PretrainedModel(data_path='data.csv', split=0.2, seed=48, batch_size = 128)
# train + save
classifierModel.train_and_save_model(pretrained=True, epochs=10, fold=3, learning_rate=1e-3, 
                path_to_pretrained_model = 'pytorch_BART', model_save_path='classifier_wt_pretrained_base_model')
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
Model.predict(csv_path = 'data.csv', batch_size = 128)