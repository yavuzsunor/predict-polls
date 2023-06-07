import os
import sys
from dataclasses import dataclass

import numpy as np
from transformers import AutoModel
from torch.utils.data import DataLoader

# import tensorflow as tf
# from tensorflow.keras.layers import Input,concatenate,Dense
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam

from src.components.data_ingestion import DataIngestionConfig
from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.logger import logging

# Load pretrained model
model = AutoModel.from_pretrained("dbmdz/distilbert-base-turkish-cased")  # .to("cuda") when GPU is accessable

@dataclass
class ModelTrainerConfig:
    model_data_path = 'artifacts/model'

class TransfomerModelLoad:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def data_loader(self, train_set, val_set):
        try:
            # Load data in batches - CAN BE RUN training_pipeline
            train_loader = DataLoader(train_set, batch_size=256, shuffle=False)
            val_loader = DataLoader(val_set, batch_size=256, shuffle=False)

            return train_loader, val_loader
        
        except Exception as e:
            raise CustomException(e, sys)

    # Get the last hidden state on top of the transformers model as the embeddings to use further in classification models 
    def get_features(data_loader):
        try: 
            for i, batch in enumerate(data_loader):
                with torch.no_grad():
                    input_ids = batch['input_ids'].to('cuda')
                    attention_mask = batch['attention_mask'].to('cuda')          
                    last_hidden_states = model(input_ids, attention_mask)
                    cls_tokens = last_hidden_states[0][:,0,:].cpu().numpy()
                    if i == 0:
                        features = cls_tokens
                    else:
                        features = np.append(features, cls_tokens, axis=0)

            return features

        except Exception as e:
            raise CustomException(e, sys)

# class DownstreamModelTrainer:

#     # Train a classification model Neural Network on top of the last hidden state in the Transfomers model
#     def train_NN_Classifier(features_tr, train_labels, features_vl, val_labels):

#         input1 = Input(shape=(features_tr.shape[1],))
#         dense1 = Dense(128,activation='relu')(input1)
#         dense2 = Dense(1,activation='sigmoid')(dense1)
#         tfmodel = Model(inputs=input1,outputs=dense2)  

#         loss = tf.keras.losses.BinaryCrossentropy (from_logits=False)

#         lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#             initial_learning_rate=1e-3,
#             decay_steps=1000,
#             decay_rate=1e-3/32)
#         optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
#         tfmodel.compile(optimizer=optimizer, loss=[loss, loss],metrics=["accuracy"])

#         checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True)
#         earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy')

#         fine_history = tfmodel.fit(features_tr, np.array(train_labels), validation_data=(features_vl, np.array(val_labels)),
#                                 epochs=10, callbacks=[checkpoint, earlystopping],batch_size=32,verbose=1)

if __name__ == "__main__":
    data_ingestion_config = DataIngestionConfig()
    train_data_path, val_data_path = data_ingestion_config.train_data_path, data_ingestion_config.val_data_path   
    
    data_transformation = DataTransformation()
    train_dataset, val_dataset = data_transformation.apply_data_transformation(train_data_path, val_data_path)

    model_trainer_config = ModelTrainerConfig()
    transfomer_model_load = TransfomerModelLoad()
    train_loader, val_loader = transfomer_model_load.data_loader(train_dataset, val_dataset)

    # # Get the last hidden state(features to fine-tune) and save them as numpy array - DONOT run without GPU
    # features_tr = transfomer_model_load.get_features(train_loader)
    # features_vl = transfomer_model_load.get_features(val_loader)   
    # np.save("train_BERT_last_hidden_states", features_tr)
    # np.save("val_BERT_last_hidden_states", features_vl)

    # Load the saved last hidden state(features to fine-tine) from artifacts
    features_tr = np.load(model_trainer_config.model_data_path + '/train_BERT_last_hidden_states.npy')
    features_vl = np.load(model_trainer_config.model_data_path + '/val_BERT_last_hidden_states.npy')

    print("train features shape", features_tr.shape)
    print("val features shape", features_vl.shape)