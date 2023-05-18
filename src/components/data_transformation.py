# Bring everything done before fine-tuning here from model_trainer.py 

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
# from sklearn.pipeline import Pipeline
from transformers import AutoTokenizer

from src.exception import CustomException
from src.logger import logging
from src.utils import MakeTensor

# Load pretrained tokenizer/model
tokenizer = AutoTokenizer.from_pretrained("dbmdz/distilbert-base-turkish-cased")


# No need to save the preprocessor as a pkl file
# @dataclass
# class DataTransformationConfig:
#     preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

@dataclass
class DataTransformation:
    # def __init__(self):
    #     self.data_transformation_config = DataTransformationConfig()

    def column_rename_clean(self, dataset):
        dataset.rename(columns = {'Tip':'Label', 'Paylaşım':'Tweets'}, inplace=True)
        dataset['Label'] = dataset['Label'].apply(lambda x: 1 if x == 'Pozitif' else 0)
        dataset.dropna(inplace=True)
        dataset.reset_index(drop=True, inplace=True)

        return dataset

    def data_columns_rename_clean_split(self, dataset):
        dataset = self.column_rename_clean(dataset)

        texts = []
        labels = []
        for i in range(len(dataset)):
            texts.append(dataset['Tweets'][i])
            labels.append(dataset['Label'][i])

        return texts, labels   

    def build_encodings(self, text):
        encodings = tokenizer.batch_encode_plus(text,
                                                max_length=128, 
                                                add_special_tokens=True, 
                                                return_attention_mask=True, 
                                                pad_to_max_length=True, 
                                                truncation=True)
        return encodings 

    def apply_data_transformation(self, train_path, val_path):

        try:
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
                          
            train_texts, train_labels = self.data_columns_rename_clean_split(train_df)
            val_texts, val_labels = self.data_columns_rename_clean_split(val_df)
            logging.info("Read train and validation data completed")
         
            train_encodings = self.build_encodings(train_texts)
            val_encodings = self.build_encodings(val_texts)

            train_dataset = MakeTensor(train_encodings, train_labels)
            val_dataset = MakeTensor(val_encodings, val_labels)
            logging.info("Applying tokenization and tensor encoding")
 
            return train_dataset, val_dataset
              
        except Exception as e:
            raise CustomException(e, sys)
    
  


















