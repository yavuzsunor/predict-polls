import sys
import os
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from zemberek import (TurkishMorphology,
                      TurkishSentenceNormalizer,
                      TurkishSpellChecker,
                      )
import snscrape.modules.twitter as sntwitter
import eksipy
import asyncio

from src.exception import CustomException
from src.utils import clean_text
from src.components.model_trainer import ModelTrainerConfig
from src.components.data_transformation import DataTransformationConfig 

morphology = TurkishMorphology.create_with_defaults()
normalizer = TurkishSentenceNormalizer(morphology)
label_sentiment = sentiment_dict = {0:"kızgın",
                                    1:"korku",
                                    2:"mutlu",
                                    3:"surpriz",
                                    4:"üzgün"}
class Inference:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.data_transformation_config = DataTransformationConfig() 
        self.saved_tokenizer = AutoTokenizer.from_pretrained(self.model_trainer_config.model_data_path)
        self.saved_model = AutoModelForSequenceClassification.from_pretrained(self.model_trainer_config.model_data_path)
    
    def scrape_twitter(self, keyword):
        try:
            scraper = sntwitter.TwitterSearchScraper(keyword, top=True)
            raw_tweets = []
            clean_tweets = []
            for i, tweet in enumerate(scraper.get_items()):
                data = [tweet.date,
                        tweet.id,
                        tweet.rawContent,
                        tweet.user.username]
                raw_tweets.append(data) #TODO: need to store the original tweets in somewhere
                clean_tweets.append(normalizer.normalize(clean_text(data)))
                if i > 2:
                    break

            print(raw_tweets, "\n")
            print("\n\n")
            print(clean_tweets, "\n")
        
        except Exception as e:
            raise CustomException(e, sys) 

    def scrape_eksisozluk(self, search_topic, entry_list=[]):
        search_topic = search_topic
        try:
            async def getTopic():
                eksi = eksipy.Eksi()
                topic = await eksi.getTopic(search_topic)
                for page in range(810, 813):
                    entrys = await topic.getEntrys(page=page)
                    for entry in entrys:
                        # print("*" * 10)
                        # print(entry.text())
                        entry_list.append(entry.text())
                #         print(entry.author.nick)
                #         print(entry.date)

                    # print("*" * 10)

            loop = asyncio.get_event_loop()
            loop.run_until_complete(getTopic())

            return entry_list

        except Exception as e:
            raise CustomException(e, sys)

    def predict_sentiment(self, tweets, model):
        try:
            encodings = self.saved_tokenizer.batch_encode_plus(tweets,
                                                                max_length=64,
                                                                add_special_tokens=True,
                                                                return_attention_mask=True,
                                                                pad_to_max_length=True,
                                                                truncation=True)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()

            test_inputs = torch.tensor(encodings["input_ids"]).to(device)
            test_masks = torch.tensor(encodings["attention_mask"]).to(device)
            with torch.no_grad():
                predictions = model(test_inputs, attention_mask=test_masks)
            
            logits = predictions[0].detach().numpy()
            return np.argmax(logits, axis=1).flatten()

        except Exception as e:
            raise CustomException(e, sys)
    
    def predict_at_inference(self, search_topic=None):
        try:
            eksi_entries = self.scrape_eksisozluk(search_topic)
            clean_eksi_entries = []
            for entry in eksi_entries:
                clean_entry = normalizer.normalize(clean_text(entry))
                # print(clean_entry, "\n")        
                clean_eksi_entries.append(clean_entry)
            
            predicted_labels = self.predict_sentiment(clean_eksi_entries, self.saved_model)

            count_sentiment = {"angry": 0, 
                            "scared": 0, 
                            "happy": 0,
                            "surprised": 0,
                            "sad":0}

            for label in list(predicted_labels):
                if label == 0:
                    count_sentiment['angry'] += 1
                elif label == 1:
                    count_sentiment['scared'] += 1
                elif label == 2:
                    count_sentiment['happy'] += 1
                elif label == 3:
                    count_sentiment['surprised'] += 1
                elif label == 4:
                    count_sentiment['sad'] += 1    

            ratio_sentiment = { k: count_sentiment[k]/sum(count_sentiment[k] for k in count_sentiment) for k in count_sentiment }
            
            return ratio_sentiment
        
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":

    inference = Inference() 
    response_json = inference.predict_at_inference("mauro icardi")
    print(response_json)