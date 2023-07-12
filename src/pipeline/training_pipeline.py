import sys
import pandas as pd

from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerConfig, ClassifierModelFinetune

class RunModelTraining:
   
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()    
        self.data_transformation = DataTransformation()
        self.classifier_model_finetune = ClassifierModelFinetune()

    def read_transform_data(self):
        try:
            df = pd.read_excel(self.model_trainer_config.raw_data_path + '/TurkishTweets.xlsx')
            tweets, labels = self.data_transformation.convert_df_clean_text(df)
            tuple_encodings = self.data_transformation.build_encodings_split_train_test(tweets, labels)
            train_dataloader, validation_dataloader = self.data_transformation.convert_to_torch_apply_data_loader(*tuple_encodings)

            return train_dataloader, validation_dataloader
        
        except Exception as e:
            raise CustomException(e, sys)

    def run_model_training(self, train_dataloader, validation_dataloader):
        try:
            trained_model = self.classifier_model_finetune.train_model(train_dataloader, validation_dataloader)
            #TODO: need to save the model
        
        except Exception as e:
            raise CustomException(e, sys)