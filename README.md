# Public Opinion Measure(Polling) using online media
Public opinion measuring(polling) has been a helpful tool for decision makers within many areas from politics to sports. The latest shift in the mediums people expressing their opinions and the latest advances in large language models with the use of transformer architecture have made it possible to come up with more accurate and fast public opinion measuring techniques with the help of web scraping and fine-tuning LLM's. 

## The problem set and proposed solution
### The problem set 
The project has been stemmed from the inaccurate pollings made by all of the polling organizations in Turkey for the last parliment and presidential election. The exploratatory research showed that the current methodologies and tools used in opinion polling not only in Turkey but overall in many countries nowadays mostly fail to predict the result of the election - sometimes by large margins. 

The idea that has been tested in this project is to scrape real-time online media posts(twitter, threads, reddit, eksisozluk(reddit-like collaborative hypertext dictionary in Turkish) to apply sentiment analysis using large language models(LLM) for certain public/political figures. 
For the 1st phase of this project, only Eksisozluk entries has been scraped and used due to the recent challenges with the Twitter API.   

### The proposed Solution
The recent research in fine-tuning and prompt engineering with the help of open-source developments have made it possible to fine-tune foundational models for downstream tasks. Here, two different 

### Experiments 
Two fine-tunings methodologies have been tested and compared for the 1st phase. 

1- Pre-trained DistilBERT model has been loaded without applying any fine-tuning...

```python
model = AutoModel.from_pretrained("dbmdz/distilbert-base-turkish-cased").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("dbmdz/distilbert-base-turkish-cased")

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

input1 = Input(shape=(features_tr.shape[1],))
dense1 = Dense(128,activation='relu')(input1)
dense2 = Dense(1,activation='sigmoid')(dense1)
tfmodel = Model(inputs=input1,outputs=dense2)
```
2- BERT Classifier model has been fine-tuned .....


### Results
Although the first method is used for a more generic sentiment classification(positive vs negative), it has been qucikly dismissed due to its poor performance comparing the second method that has used the BERTClassifier model for a multi-class downstream task fien-tuning its weigths for the custom sentiment data.


The accuracy reached with training a downstream classifier model on top of the last hidden states of DistilBERT:   
```
Epoch 1/10
226/226 [==============================] - 4s 10ms/step - loss: 0.5877 - accuracy: 0.6911 - val_loss: 0.5655 - val_accuracy: 0.7325
Epoch 2/10
226/226 [==============================] - 3s 11ms/step - loss: 0.5568 - accuracy: 0.7318 - val_loss: 0.5619 - val_accuracy: 0.7371
Epoch 3/10
226/226 [==============================] - 2s 10ms/step - loss: 0.5549 - accuracy: 0.7333 - val_loss: 0.5608 - val_accuracy: 0.7332
```

The accuracy reached with fine-tuning BERT Classifier weights with the custom data for multi-class sentiment analysis:
```
======== Epoch 1 / 5 ========
Training...
  Average training loss: 1.34
  Training epoch took: 0:00:35
Running Validation...
  Accuracy: 0.93
  Validation took: 0:00:03

======== Epoch 2 / 5 ========
Training...
  Average training loss: 0.17
  Training epoch took: 0:00:32
Running Validation...
  Accuracy: 0.99
  Validation took: 0:00:03

======== Epoch 3 / 5 ========
Training...
  Average training loss: 0.03
  Training epoch took: 0:00:32
Running Validation...
  Accuracy: 0.99
  Validation took: 0:00:03
  
======== Epoch 4 / 5 ========
Training...
  Average training loss: 0.01
  Training epoch took: 0:00:32
Running Validation...
  Accuracy: 0.99
  Validation took: 0:00:03

======== Epoch 5 / 5 ========
Training...
  Average training loss: 0.01
  Training epoch took: 0:00:32
Running Validation...
  Accuracy: 0.99
  Validation took: 0:00:03
```

## Inference  
...


## The repo structure
```
src
├── __init__.py
├── components
│   ├── __init__.py
│   ├── data_transformation.py
│   └── model_trainer.py
├── exception.py
├── logger.py
├── pipeline
│   ├── __init__.py
│   ├── inference_pipeline.py
│   └── training_pipeline.py
└── utils.py
artifacts
├── data
│   ├── tweets_binary_label
│   │   ├── data.csv
│   │   ├── train.csv
│   │   └── val.csv
│   └── tweets_multiple_label
│       └── TurkishTweets.xlsx
└── model
    ├── BERT_multi_classifier
    │   └── bert-classifier-turkish-sentiment
    │       ├── config.json
    │       ├── pytorch_model.bin
    │       ├── special_tokens_map.json
    │       ├── tokenizer.json
    │       ├── tokenizer_config.json
    │       └── vocab.txt
    └── last_hidden_state_DistilBERT
        ├── train_BERT_last_hidden_states.npy
        └── val_BERT_last_hidden_states.npy
notebook
├── Sentiment_Twitter_Turkish.ipynb
├── Sentiment_Twitter_Turkish_Classifier.ipynb
└── data
    ├── newcsv.csv
    └── sentimentSet.csv
requirements.txt
templates
├── index.html
└── home.html
app.py
README.md
requirements.txt
setup.py
 ```   

