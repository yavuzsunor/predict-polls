import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    # Load pretrained model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig() 

    def convert_df_clean_text(self, dataframe):

        try:
            dataframe.rename(columns = {'Etiket':'Label'}, inplace=True)
            dataframe['Label'] = np.where(dataframe['Label'] == 'kızgın', 0,
                                        np.where(dataframe['Label'] == 'korku', 1,
                                            np.where(dataframe['Label'] == 'mutlu', 2,
                                                np.where(dataframe['Label'] == 'surpriz', 3, 4))))
            dataframe.dropna(inplace=True)
            dataframe.reset_index(drop=True, inplace=True)
            
            # Get the lists of tweets and their labels.
            tweets = dataframe.Tweet.values
            labels = dataframe.Label.values

            return tweets, labels
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def build_encodings_split_train_test(self, tweets, labels):
        try:
            # build indices using batch_encode_plus
            indices=self.data_transformation_config.tokenizer.batch_encode_plus(list(tweets),
                                                max_length=64,
                                                add_special_tokens=True,
                                                return_attention_mask=True,
                                                pad_to_max_length=True,
                                                truncation=True)
            input_ids=indices["input_ids"]
            attention_masks=indices["attention_mask"]
            print(input_ids[0])
            print(tweets[0])

            # Use 80% for training and 20% for validation.
            train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, 
                                                                                                labels,
                                                                                                random_state=42, 
                                                                                                test_size=0.2)
            train_masks, validation_masks, _, _ = train_test_split(attention_masks, 
                                                                    labels,
                                                                    random_state=42, 
                                                                    test_size=0.2)

            logging.info("Applying tokenization, encoding and splitting data to train and valiation")
            return train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks

        except Exception as e:
            raise CustomException(e, sys) 

    def convert_to_torch_apply_data_loader(self, 
                                           train_inputs, 
                                           validation_inputs,
                                           train_labels,
                                           validation_labels,
                                           train_masks,
                                           validation_masks
                                           ):
        try:
            # Convert all of our data into torch tensors, the required datatype for our model
            train_inputs = torch.tensor(train_inputs)
            validation_inputs = torch.tensor(validation_inputs)
            train_labels = torch.tensor(train_labels, dtype=torch.long)
            validation_labels = torch.tensor(validation_labels, dtype=torch.long)
            train_masks = torch.tensor(train_masks, dtype=torch.long)
            validation_masks = torch.tensor(validation_masks, dtype=torch.long)

            batch_size = 32
            # Create the DataLoader for our training set.
            train_data = TensorDataset(train_inputs, train_masks, train_labels)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
            # Create the DataLoader for our validation set.
            validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
            validation_sampler = SequentialSampler(validation_data)
            validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

            logging.info("Converting the encodings to torch tensors and creating dataloader")
            return train_dataloader, validation_dataloader 
              
        except Exception as e:
            raise CustomException(e, sys)














