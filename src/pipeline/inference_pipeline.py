import re
from zemberek import (TurkishMorphology,
                      TurkishSentenceNormalizer,
                      TurkishSpellChecker,
                      )

morphology = TurkishMorphology.create_with_defaults()
normalizer = TurkishSentenceNormalizer(morphology)

'''
Cleaning the tweets -- THIS COULD BE PART OF data_transformation instead of doing here
'''
emoji_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+", flags = re.UNICODE)

def clean_tweet(tweet):
    if type(tweet) == np.float:
        return ""
    temp = re.sub("'", "", tweet) # to avoid removing contractions in english
    temp = re.sub("@[A-Za-z0-9iığüşöç]+","", temp)
    temp = re.sub("#[A-Za-z0-9iığüşöç]+","", temp)
    temp = re.sub(r'http\S+', '', temp)
    temp = re.sub('[()!?]', ' ', temp)
    temp = re.sub('\[.*?\]',' ', temp)
    temp = emoji_pattern.sub(r'',temp) 
#     temp = re.sub("[^a-z0-9İığüşöç]"," ", temp)
    temp = temp.split()
#     temp = [w for w in temp if not w in stopwords]
    temp = " ".join(word for word in temp)
    return temp



"""
There should be a twitter scraper code that needs to run during the inference.
It should return some results like percentage-wise positive negative ratio for a specific keyword.
It should also save the scraped tweets(maybe as csv) and their inference results(could be a json file) somewhere.
"""

# Twitter Snscrape code:
import snscrape.modules.twitter as sntwitter

scraper = sntwitter.TwitterSearchScraper("#kilicdaroglu", top=True)
tweets = []
for i, tweet in enumerate(scraper.get_items()):
    data = [tweet.date,
            tweet.id,
            tweet.rawContent,
            tweet.user.username]
    tweets.append(data)
    if i > 2:
        break

print(tweets)

# Based on the above query, tweets are stored in a list in cache, 
# and then processed through:
# cleaning,
# encoding,
# and prediction

# cleaning
tweet_samples = []
for tweet in raw_tweets:
  tweet_samples.append(normalizer.normalize(clean_tweet(tweet))) 
tweet_samples

# encoding
test_encodings = tokenizer.batch_encode_plus(tweet_samples, 
                                             max_length=128, 
                                             add_special_tokens=True, 
                                             return_attention_mask=True, 
                                             pad_to_max_length=True, 
                                             truncation=True)

# prediction
for i in range(len(test_encodings['input_ids'])):
  with torch.no_grad():
    test_input = torch.tensor(test_encodings['input_ids'][i]).to('cuda')
    test_attention = torch.tensor(test_encodings['attention_mask'][i]).to('cuda')
    last_hidden_state = model(test_input[None,:], test_attention[None,:])
    cls_token = last_hidden_state[0][:,0,:].cpu().numpy()
    if i == 0:
      test_features = cls_token
    else:
      test_features = np.append(test_features, cls_token, axis=0)

predictions = lr_clf.predict_proba(test_features)

for i, text in enumerate(tweet_samples):
  print(text)
  print("prob of being positive:", predictions[i][1], '\n')