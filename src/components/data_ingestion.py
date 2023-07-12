import boto3
import os 
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    """
    This is currently redundant because we do not load any data from 'notebook' directory. 
    """    
    train_data_path: str=os.path.join('artifacts/data', "train.csv")
    val_data_path: str=os.path.join('artifacts/data', "val.csv")
    raw_data_path: str=os.path.join('artifacts/data', "data.csv")

class DataIngestion:
    """
    This is currently redundant because we do not load any data from 'notebook' directory. 
    """
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook/data/newcsv.csv', encoding='windows-1254')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train validation split initiated")
            train_set, val_set = train_test_split(df, test_size=0.35, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            val_set.to_csv(self.ingestion_config.val_data_path, index=False, header=True)
            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.val_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)