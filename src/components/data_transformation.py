import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
# from sklearn.pipeline import Pipeline
from transformers import AutoTokenizer

from src.components.data_ingestion import DataIngestion
from src.exception import CustomException
from src.logger import logging
from src.utils import MakeTensor

# Load pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained("dbmdz/distilbert-base-turkish-cased")

@dataclass
class DataTransformation:

    path_train = 'artifacts/'

    def convert_df_clean_text(self, dataframe):

        try:        
            dataframe.rename(columns = {'Tip':'Label', 'Paylaşım':'Tweets'}, inplace=True)
            dataframe['Label'] = dataframe['Label'].apply(lambda x: 1 if x == 'Pozitif' else 0)
            dataframe.dropna(inplace=True)
            dataframe.reset_index(drop=True, inplace=True)
            
            texts = []
            labels = []
            for i in range(len(dataframe)):
                texts.append(dataframe['Tweets'][i])
                labels.append(dataframe['Label'][i])

            return texts, labels
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def build_encodings(self, text):
        try:
            encodings = tokenizer.batch_encode_plus(text,
                                                    max_length=128, 
                                                    add_special_tokens=True, 
                                                    return_attention_mask=True, 
                                                    pad_to_max_length=True, 
                                                    truncation=True)
            return encodings

        except Exception as e:
            raise CustomException(e, sys) 

    def apply_data_transformation(self, train_path, val_path):

        try:
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
                          
            train_texts, train_labels = self.convert_df_clean_text(train_df)
            val_texts, val_labels = self.convert_df_clean_text(val_df)
            logging.info("Read train and validation data completed")
         
            train_encodings = self.build_encodings(train_texts)
            val_encodings = self.build_encodings(val_texts)

            train_dataset = MakeTensor(train_encodings, train_labels)
            val_dataset = MakeTensor(val_encodings, val_labels)
            logging.info("Applying tokenization and tensor encoding")

            print("input_ids dimensions", np.array(train_dataset.encodings['input_ids']).shape, '\n')
            print("attention_mask dimensions", np.array(train_dataset.encodings['attention_mask']).shape, '\n')

            return train_dataset, val_dataset
              
        except Exception as e:
            raise CustomException(e, sys)


















