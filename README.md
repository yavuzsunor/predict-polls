# Public Opinion Measure(Polling) using online media with fine-tuning BERT model
Public opinion measuring(polling) has been a helpful tool for decision makers within many areas from politics to sports. The latest shift in the mediums people expressing their opinions and the latest advances in large language models with the use of transformer architecture have made it possible to come up with more accurate and fast public opinion measuring techniques with the help of web scraping and fine-tuning LLM's. 

## The problem set and proposed solution
### The problem set 
The project has been stemmed from the inaccurate pollings made by all of the polling organizations in Turkey for the last parliamentary and presidential election. The exploratory research showed that the current methodologies and tools used in opinion polling not only in Turkey but overall in many countries nowadays mostly fail to predict the result of the elections - sometimes by large margins. 

The idea that has been tested in this project is to scrape real-time online media posts(twitter, threads, reddit, eksisozluk(reddit-like collaborative hypertext dictionary in Turkish) to apply sentiment analysis using large language models(LLM) for certain public/political figures. 
For the 1st phase of this project, only Eksisozluk entries have been scraped and used due to the recent challenges with the Twitter API.   

### The proposed Solution
The recent research in fine-tuning and prompt engineering with the help of open-source developments have made it possible to fine-tune foundational models for downstream tasks. Here, two different methodologies have been studied and tested at real-time inference.  

### Experiments 
Two fine-tunings methodologies have been tested and compared for the 1st phase. 

1- Using the pre-trained DistilBERT model without applying any fine-tuning to fetch the last hidden states - embeddings - from the top layer and train a custom NN classifier using those embeddings as the feature set:   

```python
model = AutoModel.from_pretrained("dbmdz/distilbert-base-turkish-cased").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("dbmdz/distilbert-base-turkish-cased")
```
```python
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
```
```python
input1 = Input(shape=(features_tr.shape[1],))
dense1 = Dense(128,activation='relu')(input1)
dense2 = Dense(1,activation='sigmoid')(dense1)
tfmodel = Model(inputs=input1,outputs=dense2)
```

2- Using the BERT Classifier model to fine-tune the model weights using a custom multi-labeled data to train a multiclassifier for the downstream task:

```python
sentiment_dict = {"kızgın":0,  # angry
                  "korku":1,   # feared/threatened
                  "mutlu":2,   # happy
                  "surpriz":3, # surprised
                  "üzgün":4}.  # sad
plot_grams(process_text(dataset_tweets, label=2, gram='bi'), sentiment='mutlu')
```
![alt text](/images/bigrams_mutlu.png)

```python
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
```
- > site ne zaman çalıştıda ürün stokları bitti diyor mal mısınız oğlum kimi kandırıyorsunuz!
    ```python
    Tokenized:  ['site', 'ne', 'zaman', 'çalıştı', '##da', 'ürün', 'stok', '##ları', 'bitti', 'diyor', 'mal', 'mısınız', 'oğlum', 'kimi', 'kandır', '##ıyorsunuz', '!']
    Token IDs:  [6521, 2142, 2248, 9879, 1986, 2782, 10992, 2037, 10638, 4022, 2810, 12760, 9747, 7470, 13413, 16566, 5]
    ```
- > Sebebi neydi ki diye bağıracağım şimdi az kaldı
    ```python
    Tokenized:  ['Seb', '##ebi', 'neydi', 'ki', 'diye', 'bağır', '##acağım', 'şimdi', 'az', 'kaldı']
    Token IDs:  [9985, 9254, 21074, 2402, 2636, 7355, 6950, 3653, 2539, 5234]
    ```

```python
from transformers import AutoModelForSequenceClassification, AdamW, AutoConfig
config = AutoConfig.from_pretrained(
        "dbmdz/bert-base-turkish-cased",num_labels=5)
# Load BertForSequenceClassification, the pretrained BERT model with a single
# linear classification layer on top.
model = AutoModelForSequenceClassification.from_pretrained(
    "dbmdz/bert-base-turkish-cased",config=config)
```

### Results
Although the first method is used for a more generic sentiment classification(positive vs negative), it has been quickly dismissed due to its poor performance comparing the second method that has used the BERTClassifier model for a multi-class downstream task fine-tuning its weigths for the custom sentiment data.


The accuracy reached with training a downstream classifier model on top of the last hidden states of DistilBERT:   
```
...

Epoch 3/10
226/226 [==============================] - 2s 10ms/step - loss: 0.5549 - accuracy: 0.7333 - val_loss: 0.5608 - val_accuracy: 0.7332
```

The accuracy reached with fine-tuning BERT Classifier weights with the custom data for multi-class sentiment analysis:
```
...

======== Epoch 5 / 5 ========
Training...
  Average training loss: 0.01
  Training epoch took: 0:00:32
Running Validation...
  Accuracy: 0.99
  Validation took: 0:00:03
```

## Inference  
To test the model, the recent public opinion was measured for two public figures, a Turkish soccer prodigy "Arda Guler" who just made a sensational transfer to Real Madrid and the opposition party leader "Kemal Kilicdaroglu" who just lost the presidential election a month ago. As can be seen from the below results, Arda Guler has had mainly a positive sentiment in the public with some mixed surprising and angry feelings - most likely due to the excitement and disappointment of his former club's fans - and Kemal Kilicdaroglu had caused mostly anger and sadness in the public especially within the voters who had been dissapointed with his performance in the last election.       


json payload for Arda Guler's public sentiment:
```json
{'angry': 0.27, 
 'scared': 0.03, 
 'happy': 0.53, 
 'surprised': 0.13, 
 'sad': 0.03}
```

json payload for Kemal Kilicdaroglu's public sentiment:
```json
{'angry': 0.45, 
 'scared': 0.05, 
 'happy': 0.15, 
 'surprised': 0.20, 
 'sad': 0.15}
```

The fine-tuned model can also be tested and downloaded at the follwing Hugginfgace repo:
https://huggingface.co/sunor/bert-classifier-turkish-sentiment

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

