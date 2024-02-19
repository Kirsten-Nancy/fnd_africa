import re
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from deep_translator import GoogleTranslator

# df = pd.read_csv('data//africa_news_tweets.csv')
df = pd.read_csv('data/afndet.csv')
print(df.head())
# print(df['Label'].value_counts())
# print(df.info())

# Data Preprocessing
# Remove urls, non-alphanumeric chars,emojis
def remove_emojis(text):
    emoji_pattern = re.compile(
            u'(\U0001F1F2\U0001F1F4)|'       # Macau flag
            u'([\U0001F1E6-\U0001F1FF]{2})|' # flags
            u'([\U0001F600-\U0001F64F])'     # emoticons
            "+", flags=re.UNICODE)
    
    return emoji_pattern.sub('', text)


def preprocessing(text):
    text = remove_emojis(text)
    text = re.sub(r"\d+", '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove urls
    text = re.sub(r'http\S+', '', text)
    text = text.lower()
    translated_text = GoogleTranslator(source='en', target='sw').translate(text)
    print('translated text', translated_text)
    return translated_text

print('starrt')
df.loc[:, "News"] = df['News'].apply(preprocessing)
print(df.head(10))

df.to_csv('data//afndet_sw.csv', index=False)
