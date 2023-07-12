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
test_tweets = [
    'Özgür ve temiz bir turkiye icin 🇹🇷\n#SanaSoz \n#SanaSozBaharlarGelecek \n#cezevindenseskaydi \n#bizdendesanasoz \n#kilicdaroglu \n#GenelAf \n#infazduezenlemesi \n#ADALET \n#af \n#MilletTarihYazacak \n#Milletİttifakı https://t.co/9hhKFNftGs',
    'İşte bu! Türkiye’me yeni bir nefes #kılıçdaroğlu https://t.co/APxXlsvV4l',
    'MüslümanızElhamdürillah\nkalkıp da terörisİTlerin desteklediği yalamalara OY verirmiyiz\nkatiLLErin ittifakçılarına OY verirmiyiz\nİnsanız Biz,beynimiz var Allah akılfikir ihsan eylemiş\nanguSdeğilik\nKuranıkerim yakan bebek katillerine OY istiyor kasetci #Kılıçdaroğlu  ve adaylarda👇 https://t.co/n5hAsxKRqb',
    'Şimdi #HaberGlobal; İçişleri bakanı #Soylu, \n-#Demirtaş\'ın #PKK\'ya "silah bıraktırma" çıkışı örgüte nefeslenme numarasıdır, diyor.\nBu doğru değil..\nPKK ilk defa #Sevr\'deki bölünme projesi\'ni #Kılıçdaroğlu ve #Milletittifakı yapılanmasıyla güçlü olrk yakalamıştır, buna oynuyor https://t.co/ppiazSatHI',
    'İhanetin arkasında #emperyalizm ile işbirliği yapan #Türksolu var.\n-09 temmuz 1934 1.Kürdoloji kongresi"\n-1955 Moskova Komünist yazarlar kongresi\n-1978 Lice, Fis köyü #PKK\'nın kuruluşu.\nBu iğrenç sol, #Kılıçdaroğlu ile harekete geçti.. https://t.co/feD6TwpwiF',
    'Ne diyor #Kılıçdaroğlu; "#Kürtkimliği\'ni meclise kabul ettireceğim, #yerelyönetimler yasasını çıkaracağım". Bu sinsi bir Sevr çıkışıdır\n#Sevr\'i imzlayanlar "vatan haini" ilan edilmiştir.\nBu ihaneti #Atatürk ve arkadaşları "Kurtuluş savaşı" sonrası #Lozan\'da yırtıp atmışlardır. https://t.co/JJJWQ2M8kW',
    'Kılıçdaroğlu Bu Sefer Çok Güçlü. #kılıçdaroğlu #chp https://t.co/JgSoLgI1da @YouTube aracılığıyla',
    "@__KESAFET64__ #Kılıçdaroğlu’na oy yok👎🏻",
    '#MuhammetYakut #Tether\n#kilicdaroglu #ibbguvenligizambekliyor %4 zam %300 enflasyon bu gidiş nereye',
    "Kılıçdaroğlu 'Alevi' notuyla paylaştığı videoda ilk kez oy kullanacak gençlere seslendi \nhttps://t.co/nFj6WBQhxT \n#kılıçdaroğlu #alevi https://t.co/l63u311gYD",
    'Optimar, İlk kez oy kullanacak gençlerin nabzının tutulduğu anketinin sonuçlarını yayınladı:   📷 % 51,2 Erdoğan,  📷 % 39,2 Kılıçdaroğlu   📷 % 7,4 Muharrem İnce  📷 % 1,1 Sinan Oğan  \n\n#anket #optimar #seçim2023 #erdoğan #kılıçdaroğlu #ince #oğan https://t.co/NE2ORt22E4',
]


class Inference:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.data_transformation_config = DataTransformationConfig() 
        self.saved_tokenizer = AutoTokenizer.from_pretrained(self.model_trainer_config.model_data_path)
        self.saved_model = AutoModelForSequenceClassification.from_pretrained(self.model_trainer_config.model_data_path)
    
    def scrape_twitter(self, keyword):
        """
        There should be a twitter scraper code that needs to run during the inference.
        It should return some results like percentage-wise positive negative ratio for a specific keyword.
        It should also save the scraped tweets(maybe as csv) and their inference results(could be a json file) somewhere.
        """
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
                for page in range(6171, 6173):
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

if __name__ == "__main__":

    inference = Inference()
    # inference.scrape_twitter('#kilicdaroglu') 
    eksi_entries = inference.scrape_eksisozluk('kemal kılıçdaroğlu')

    clean_eksi_entries = []
    for entry in eksi_entries:
        clean_entry = normalizer.normalize(clean_text(entry))
        # print(clean_entry, "\n")        
        clean_eksi_entries.append(clean_entry)


    predicted_labels = inference.predict_sentiment(clean_eksi_entries, inference.saved_model)

    for orig_entry, final_entry, predicted_label in zip(eksi_entries, clean_eksi_entries, predicted_labels):
        # Print the original tweet
        print(orig_entry, '\n')
        # Print the final clean tweet
        print(final_entry, '\n')
        # # Print the sentence split into tokens.
        # print('Tokenized: ', inference.saved_tokenizer.tokenize(tweet))
        # # Print the sentence mapped to token ids.
        # print('Token IDs: ', inference.saved_tokenizer.convert_tokens_to_ids(inference.saved_tokenizer.tokenize(tweet)))
        print(label_sentiment[predicted_label], '\n\n')
