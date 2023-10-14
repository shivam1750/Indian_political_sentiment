# importing packages
import pandas as pd
import re
import numpy as np
import datetime as dt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
df = pd.read_csv('A:/prac_sem/SEM7/prac/mini_project/final_text_sentiment.csv', sep = '|')
print ('Output from Twitter read...')
def clean_tweet(text):
    result = re.sub(r'http\S+', '', text)
    result = re.sub('[.@#]', '', result)
    result = re.sub(r'\w+(?:/\w+)+',"", result)
    result = re.sub(r'\w+(?:-\w+)+',"", result)
    result = re.sub(" \d+", " ", result)
    result = " ".join([word for word in result.split() if len(word) > 2])
    return result
df['clean_tweet'] = df['text'].apply(clean_tweet)
print ('Tweets cleaned...')
df['sentiment'] = df['clean_tweet'].apply(analyser.polarity_scores)
print ('Sentiment scores obtained...')
def get_compound_score(dict_row):
    result = dict_row['compound']
    return result
df['sentiment_score'] = df['sentiment'].apply(get_compound_score)
def sentiment_class(score):
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    elif score == 0.0:
        return 'Neutral'
    else:
        print ('Weird response!')
df['sentiment_class'] = df['sentiment_score'].apply(sentiment_class)
print ('Saving the final output to disk...')
df = df[['date', 'clean_tweet', 'sentiment_score', 'sentiment_class']]
df.to_csv('final_text_sentiment.csv', index = False, sep = '|')