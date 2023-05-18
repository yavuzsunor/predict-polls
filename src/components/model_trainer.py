from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader

import tensorflow as tf
from tensorflow.keras.layers import Input,concatenate,Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# Load pretrained tokenizer/model
tokenizer = AutoTokenizer.from_pretrained("dbmdz/distilbert-base-turkish-cased")
model = AutoModel.from_pretrained("dbmdz/distilbert-base-turkish-cased")  # .to("cuda") when GPU is accessable


# TODO - need to import data transformation to get train_text, val_text, train_labels, val_labels


def build_encodings(text):
    encodings = tokenizer.batch_encode_plus(text,
                                            max_length=128, 
                                            add_special_tokens=True, 
                                            return_attention_mask=True, 
                                            pad_to_max_length=True, 
                                            truncation=True)
    return encodings

# Run the encodings - CAN BE RUN training_pipeline  
train_encodings = build_encodings(train_text)
val_encodings = build_encodings(val_text)


class MakeTensor(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Run MakeTensor - CAN BE RUN training_pipeline
train_dataset = MakeTensor(train_encodings, train_labels)
val_dataset = MakeTensor(val_encodings, val_labels)

# Load data in batches - CAN BE RUN training_pipeline
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# Get the last hidden state on top of the transformers model as the embeddings to use further in classification models 
def get_features(data_loader):
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


# Train a classification model Neural Network on top of the last hidden state in the Transfomers model
def train_NN_Classifier(features_tr, train_labels, features_vl, val_labels):

    input1 = Input(shape=(features_tr.shape[1],))
    dense1 = Dense(128,activation='relu')(input1)
    dense2 = Dense(1,activation='sigmoid')(dense1)
    tfmodel = Model(inputs=input1,outputs=dense2)  

    loss = tf.keras.losses.BinaryCrossentropy (from_logits=False)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=1000,
        decay_rate=1e-3/32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    tfmodel.compile(optimizer=optimizer, loss=[loss, loss],metrics=["accuracy"])

    checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True)
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy')

    fine_history = tfmodel.fit(features_tr, np.array(train_labels), validation_data=(features_vl, np.array(val_labels)),
                            epochs=10, callbacks=[checkpoint, earlystopping],batch_size=32,verbose=1)

    # need to return model object to save in artifacts