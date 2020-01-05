import re
from textblob import TextBlob
import pandas as pd


cols = ['id','date','user','text']
collected_data = pd.read_csv(r"C:\Users\User\PycharmProjects\Sentiment_Analysis_Twitter_Data\raw_data.csv",header=None, names=cols)


def clean_text(text):
    """
    Utility function to clean tweet text by removing links, special characters
    using simple regex statements.
    """
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())

collected_data['clean_data'] = collected_data.text.apply(clean_text)

def text_polarity(text):
    return TextBlob(text).sentiment.polarity

collected_data['polarity'] = collected_data.clean_data.apply(text_polarity)


def tweet_sentiment(text):
    '''
    Utility function to classify sentiment of passed tweet
    using textblob's sentiment method
    '''
    analysis=TextBlob(text).sentiment.polarity
    # set sentiment
    if analysis > 0:
        return 'positive'
    elif analysis == 0:
        return 'neutral'
    else:
        return 'negative'



collected_data['sentiments'] = collected_data.clean_data.apply(tweet_sentiment)

#dataframe with polarity , sentiments column added
print(collected_data.head())

df=pd.get_dummies(collected_data['sentiments'])
collected_data=pd.concat([collected_data,df],axis=1)

#Total number of Each type of tweets
ptweets=collected_data['positive'].sum()
ntweets=collected_data['negative'].sum()
tweets=ptweets+ntweets
print(ptweets,ntweets,tweets)


# percentage of positive tweets
print("Positive tweets percentage: {} %".format(100 * (ptweets) / (tweets)))
# percentage of negative tweets
print("Negative tweets percentage: {} %".format(100 * (ntweets) / (tweets)))



#collected_data.to_csv('sentiments_data.csv',encoding='utf-8')
