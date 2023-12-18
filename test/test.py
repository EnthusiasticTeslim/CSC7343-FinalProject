



# # ************************ How to train the base model ************************
# from project import Train_Base_PretrainedModel
# # create an instance of the model
# classifierModel = Train_Base_PretrainedModel(data_path='data.csv', split=0.2, seed=48, batch_size = 128)
# # train + save
# # train_and_save_model(self, pretrained=False,  epochs=10, fold=3, learning_rate=2e-3, path_to_pretrained_model = None, model_save_path='base_model'):
# #classifierModel.train_and_save_model(pretrained=False, epochs=1, fold=3, learning_rate=1, model_save_path='classifier_wt_base_model')
# classifierModel.train_and_save_model(pretrained=True, epochs=1, fold=3, learning_rate=1, 
#                 path_to_pretrained_model = '../trainedBART_epochs50/reformed_pytorch_model_BART', model_save_path='classifier_wt_pretrained_base_model')
# # calculate time
# classifierModel.calculate_total_time()

# ************************ How to train the classifier ************************
# from project import PreTrainer
# pretrainer = PreTrainer(data_path='data.csv', epochs=1, mlm_probability=0.15)
# pretrainer.train(method = 'BART') # method = 'BERT' or 'BART', default is 'BART', you can change it to 'BERT' if you want to pretrain a BERT model
# pretrainer.get_pytorch_version(new_model_name ='pytorch_BART' )

# ************************ Test Trained Model ************************
from project import ModelTrainer
# create an instance of the model
Model = ModelTrainer(load_trained_classifier=True, path_to_trained_classifier='../model/BARTpretrained_model_sum_balanced_epoch10_seed48_classified_V3')
# apply the model on the test data
Model.predict(csv_path = '../data/data_balanced.csv', batch_size = 128)