import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import WordPunctTokenizer
import re
from bs4 import BeautifulSoup
plt.style.use('fivethirtyeight')

cols = ['id','date','user','text','clean_data','polarity','sentiments','negative','neutral','positive']
train = pd.read_csv(r"C:\Users\User\PycharmProjects\Sentiment_Analysis_Twitter_Data\sentiments_data.csv",header=None, names=cols)

train['sentiment']=train['sentiments'].map({'negative':0,'positive':4,'neutral':2})
train.drop(['sentiments','polarity','id','date','user','positive','negative','neutral'],axis=1,inplace=True)
train['pre_clean_len'] = [len(t) for t in train.text]

#print(train.head())



fig, ax = plt.subplots(figsize=(5, 5))
plt.boxplot(train.pre_clean_len)
plt.show()

tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9_]+'
pat2 = r'https?://[^ ]+'
combined_pat = r'|'.join((pat1, pat2))
www_pat = r'www.[^ ]+'
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'html.parser')
    souped = soup.get_text()
    try:
        bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        bom_removed = souped
    stripped = re.sub(combined_pat, '', bom_removed)
    stripped = re.sub(www_pat, '', stripped)
    lower_case = stripped.lower()
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = [x for x  in tok.tokenize(letters_only) if len(x) > 1]
    return (" ".join(words)).strip()


nums = [0,5000]
print ("Cleaning and parsing the tweets...\n")
clean_tweet_texts = []
for i in range(nums[0],nums[1]):
    if( (i+1)%1000 == 0 ):
        print ("Tweets %d of %d has been processed" % ( i+1, nums[1] ))
    clean_tweet_texts.append(tweet_cleaner(train['text'][i]))


clean_df = pd.DataFrame(clean_tweet_texts,columns=['text'])
clean_df['target'] = train.sentiment
print(clean_df.tail())

clean_df.to_csv('clean_data.csv',encoding='utf-8')


