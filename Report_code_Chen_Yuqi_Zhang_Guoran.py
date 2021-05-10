#!/usr/bin/env python
# coding: utf-8

---
**Acknowledgement**

Thanks for Guoran Zhang writing Classification and result part of this project. We did really well for our classification results.
---

# In[1]:


# Our Github link is: https://github.com/DavidChen25/Covid-19-Sentiment-Analysis

import pandas as pd
import re
import string
from nltk.corpus import stopwords 
from collections import Counter
import jieba 
import matplotlib.font_manager as fm
from PIL import Image
from wordcloud import WordCloud,ImageColorGenerator,STOPWORDS
import numpy as np
from termcolor import colored
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from matplotlib import pyplot as plt
from matplotlib import ticker
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


data_train = pd.read_csv('Corona_NLP_train.csv', encoding='latin1')
data_test = pd.read_csv('Corona_NLP_test.csv', encoding='latin1')
data_train.columns = ['ID', 'ScreenName', 'Location', 'Date', 'Tweet', 'Sentiment']
data_test.columns = ['ID', 'ScreenName', 'Location', 'Date', 'Tweet', 'Sentiment']
data_train.drop(['ID', 'Date', 'ScreenName', 'Location'], axis = 1, inplace = True)
data_test.drop(['ID', 'Date', 'ScreenName', 'Location'], axis = 1, inplace = True)

print(data_train.dropna())
print(data_test.dropna())


# ### Pre-Processing

# In[3]:


remove_url=lambda x: re.sub(r'https\S+' , '',str(x))
data_train['Tweet'] = data_train['Tweet'].apply(remove_url)
#coverting all tweets to lowercase
to_lower=lambda x: x.lower()
data_train['Tweet'] = data_train['Tweet'].apply(to_lower)
#remove punctutation
remove_puncts=lambda x: x.translate(str.maketrans('','',string.punctuation))
data_train['Tweet'] = data_train['Tweet'].apply(remove_puncts)
#remove stopwords
more_words=['covid','covid19']
stop_words=set(stopwords.words('English'))
stop_words.update(more_words)


# In[4]:


# Function to process tweets
def clean_tweet(data, wordNetLemmatizer):
	data['Tweet'] = data['Tweet']
	data['Tweet'] = data['Tweet'].str.replace("@[\w]*","")
	data['Tweet'] = data['Tweet'].str.replace("[^a-zA-Z' ]","")
	data['Tweet'] = data['Tweet'].replace(re.compile(r"((www\.[^\s]+)|(https?://[^\s]+))"), "")
	data['Tweet'] = data['Tweet'].replace(re.compile(r"(^| ).( |$)"), " ")
	data['Tweet'] = data['Tweet'].str.split()
	data['Tweet'] = data['Tweet'].apply(lambda tweet: [word for word in tweet if word not in stop_words])
	data['Tweet'] = data['Tweet'].apply(lambda tweet: [wordNetLemmatizer.lemmatize(word) for word in tweet])
	data['Tweet'] = data['Tweet'].apply(lambda tweet: ' '.join(tweet))
	return data

# Define processing methods
wordNetLemmatizer = WordNetLemmatizer()

# Pre-processing the tweets
train_data = clean_tweet(data_train, wordNetLemmatizer)
train_data.to_csv('clean_train.csv', index = False)
test_data = clean_tweet(data_test, wordNetLemmatizer)
test_data.to_csv('clean_test.csv', index = False)

words_list=[word for line in train_data['Tweet'] for word in line.split()]
words_list[:5]
word_counts=Counter(words_list).most_common(50)
words_df=pd.DataFrame(word_counts)
words_df.columns=['word','frq']
fig = px.bar(words_df,x='word',y='frq',title='Most common words')
fig.update_xaxes(tickangle = -45)
fig.show()


# - The original common words showing above include those high frequent hashtags like “COVID”, “covid-19”, and “CORONAVIRUS” etc. We tend to remove these words and only keep “coronavirus” in the list. There are some high frequent words such like “price”, “food”, “consumer”, and “pandemic” etc.

# ### Sentiment Analysis

# - We introduce the TextBlob package to do some Sentiment Analysis visualization. It has two main metrics as polarity and subjectivity. We apply the TextBlob API onto our data to do sentiment analysis of all tweets. The two graphs are shown below:

# In[6]:


from textblob import TextBlob
train_data['polarity'] = train_data['Tweet'].apply(lambda x: TextBlob(x).sentiment.polarity)
train_data['subjectivity'] = train_data['Tweet'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# The histograms of polarity and subjectivity scores for all tweets in the dataset.
fig = plt.figure(figsize=(8, 6))
train_data['polarity'].hist()
plt.xlabel('Polarity Score', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

fig = plt.figure(figsize=(8, 6))
train_data['subjectivity'].hist()
plt.xlabel('Subjectivity Score', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Sentiment count
sns.set_style('darkgrid')
plt.figure(figsize = (8, 8))
plt.xlabel('Sentiment', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.countplot(x = train_data['Sentiment'] , palette = 'viridis')
plt.show()


# - Most of the tweets in the dataset seem to be neutral and not much subjectivity. However, in the polar tweets, there are slightly more positively charged tweets than negatively charged tweets. We can say that covid-19 pandemic on twitter is generally optimistic, but it would be nice to see what sentiment that people’s feelings and thoughts are

# ### WordCloud Map

# In[7]:


from wordcloud import WordCloud

# texts from all tweets
words = ' '.join([text for text in train_data['Tweet']])
wordcloud = WordCloud(width=800, height=500, random_state=21,  min_font_size = 10, max_font_size=110, background_color="white").generate(words)

plt.figure(figsize=(8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


# In[8]:


ExPositive_words = ' '.join([text for text in  train_data['Tweet'][ train_data['Sentiment'] == 'Extremely Positive']])

wordcloud = WordCloud(width=800, height=500, random_state=21,  min_font_size = 10, max_font_size=110, background_color="white").generate(ExPositive_words)

plt.figure(figsize=(8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


# In[9]:


ExNegative_words = ' '.join([text for text in  train_data['Tweet'][ train_data['Sentiment'] == 'Extremely Negative']])

wordcloud = WordCloud(width=800, height=500, random_state=21,  min_font_size = 10, max_font_size=110, background_color="white").generate(ExNegative_words)

plt.figure(figsize=(8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


# In[10]:


Neutral_words = ' '.join([text for text in train_data['Tweet'][ train_data['Sentiment'] == 'Neutral']])

wordcloud = WordCloud(width=800, height=500, random_state=21,  min_font_size = 10, max_font_size=110, background_color="white").generate(Neutral_words)

plt.figure(figsize=(8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


# - These word clouds yield similar result as the common words in figure 3. The extremely positive sentiment shows the reference words like “great”, “best”, “thank”, “hand-sanitizer”, and “free”. Moreover, the extremely negative sentiment shows the reference words like “panic”, “crisis”, and “fear” So I think it is more likely want to emphasis lots of people are experiencing the crisis period and feel panic and fear of the virus, but those people who get over the virus and get recovered are more optimal like they feel more thankful to the doctors or those vaccines that helps them recover.

# ### Vectorization

# In[15]:


# Tf-IDF (Term Frequency — Inverse Document Frequency)
tfidfVectorizer = TfidfVectorizer(min_df = 5, max_features = 1000)
tfidfVectorizer.fit(train_data['Tweet'].apply(lambda x: np.str_(x)))
train_tweet_vector = tfidfVectorizer.transform(train_data['Tweet'].apply(lambda x: np.str_(x)))
test_tweet_vector = tfidfVectorizer.transform(test_data['Tweet'].apply(lambda x: np.str_(x)))

# part-of-speech taggings 
word_tokenized = sent_tokenize(str(train_data['Tweet']))
for i in word_tokenized:
    wordsList = nltk.word_tokenize(i)
    wordsList = [w for w in wordsList if not w in stop_words] 
    pos_tag = nltk.pos_tag(wordsList)
    print(pos_tag)


# In[14]:


# N-grams 

# The following part of code is from Ayush Pareek's Github: "https://github.com/ayushoriginal/Sentiment-Analysis-Twitter"
# I generated the unigram + negation handling feature but do not know how to fit the model. The part of uncomplete feature 
# builded ode is show in the last comment.

import nltk
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.metrics import BigramAssocMeasures
from nltk.collocations import  BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import  BigramAssocMeasures

def unigrams(tweets, add_negtn_feat): 
    unigrams_fd = nltk.FreqDist()

    for words in tweets:
        words_uni = words
        unigrams_fd.update(words)

    #unigrams_sorted = nltk.FreqDist(unigrams).keys()
    unigrams_sorted = unigrams_fd.keys()
    mostcommon=unigrams_fd.most_common(50)
    #bigrams_sorted = nltk.FreqDist(bigrams).keys()
    #trigrams_sorted = nltk.FreqDist(trigrams).keys()
       
    def get_word_features(words):
        bag = {}
        words_uni = [ 'has(%s)'% ug for ug in words ]
        for f in words_uni:
            bag[f] = 1
        print(bag)
        #bag = collections.Counter(words_uni+words_bi+words_tri)
        return bag
    
    negtn_regex = re.compile( r"""(?:
        ^(?:never|no|nothing|nowhere|noone|none|not|
            havent|hasnt|hadnt|cant|couldnt|shouldnt|
            wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint
        )$
    )
    |
    n't
    """, re.X)

    def get_negation_features(words):
        INF = 0.0
        negtn = [ bool(negtn_regex.search(w)) for w in words ]
    
        left = [0.0] * len(words)
        prev = 0.0
        for i in range(0,len(words)):
            if( negtn[i] ):
                prev = 1.0
            left[i] = prev
            prev = max( 0.0, prev-0.1)
    
        right = [0.0] * len(words)
        prev = 0.0
        for i in reversed(range(0,len(words))):
            if( negtn[i] ):
                prev = 1.0
            right[i] = prev
            prev = max( 0.0, prev-0.1)
    
        return dict( zip(
                        ['neg_l('+w+')' for w in  words] + ['neg_r('+w+')' for w in  words],
                        left + right ) )
    
    def extract_features(words):
        features = {}

        word_features = get_word_features(words)
        features.update( word_features )

        if add_negtn_feat:
            negation_features = get_negation_features(words)
            features.update( negation_features )
 
        
        return features
    a=[]
    for sentence in tweets:
        feature=extract_features(sentence)
        a.append(feature)
    return a, unigrams_sorted, mostcommon
features, unigrams_sorted, mostcommon = unigrams(['good good good not good'.split()], True)
print(features)
print(unigrams_sorted)
print(mostcommon)

def text():
    strg=''
    for sentence in new_sentence:
        temp=" ".join(sentence)
        strg+=temp
    print(string)
    return string

def bag_of_words():
    feature={}
    for sentence in new_sentence: 
        temp= dict([(word,True) for word in sentence])
        feature.update(temp)
    return feature 

def  bigram(words,score_fn=BigramAssocMeasures.chi_sq,n=1000):
    bigram_finder=BigramCollocationFinder.from_words(words)
    bigrams= bigram_finder.nbest(score_fn,n)
    newBigrams = [u+v for (u,v) in bigrams]
    return bag_of_words(newBigrams) 

def build_features():
    #feature = bag_of_words()
    #feature = bigram(text(),score_fn=BigramAssocMeasures.chi_sq,n=500)
    print(feature) 

# The results are hard to read.
# I tried these functions that I want to extract those features by part but I still do not know how to combine those features 
# and fit the model


# ### Sentiment 

# In[16]:


from sklearn.model_selection import train_test_split

train,valid = train_test_split(train_data,test_size = 0.2,random_state=0,stratify = train_data.Sentiment.values) #stratification means that the train_test_split method returns training and test subsets that have the same proportions of class labels as the input dataset.
print("train shape : ", train.shape)
print("valid shape : ", valid.shape)

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

stop = list(stopwords.words('english'))
vectorizer = CountVectorizer(decode_error = 'replace',stop_words = stop)

X_train = vectorizer.fit_transform(train.Tweet.values)
X_valid = vectorizer.transform(valid.Tweet.values)

y_train = train.Sentiment.values
y_valid = valid.Sentiment.values

print("X_train.shape : ", X_train.shape)
print("X_train.shape : ", X_valid.shape)
print("y_train.shape : ", y_train.shape)
print("y_valid.shape : ", y_valid.shape)    
    


# ### Classification Models

# In[19]:


# Naive Bayes Classifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

naiveByes_clf = MultinomialNB()

naiveByes_clf.fit(X_train,y_train)

NB_prediction = naiveByes_clf.predict(X_valid)
NB_accuracy = accuracy_score(y_valid,NB_prediction)
print("training accuracy Score    : ",naiveByes_clf.score(X_train,y_train))
print("Validation accuracy Score : ",NB_accuracy )
print(classification_report(NB_prediction,y_valid))
    

# Stochastic Gradient Descent-SGD Classifier
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(loss = 'hinge', penalty = 'l2', random_state=0)

sgd_clf.fit(X_train,y_train)

sgd_prediction = sgd_clf.predict(X_valid)
sgd_accuracy = accuracy_score(y_valid,sgd_prediction)
print("Training accuracy Score    : ",sgd_clf.score(X_train,y_train))
print("Validation accuracy Score : ",sgd_accuracy )
print(classification_report(sgd_prediction,y_valid))


# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier()

rf_clf.fit(X_train,y_train)

rf_prediction = rf_clf.predict(X_valid)
rf_accuracy = accuracy_score(y_valid,rf_prediction)
print("Training accuracy Score    : ",rf_clf.score(X_train,y_train))
print("Validation accuracy Score : ",rf_accuracy )
print(classification_report(rf_prediction,y_valid))


# Extreme Gradient Boosting
import xgboost as xgb

xgboost_clf = xgb.XGBClassifier()

xgboost_clf.fit(X_train, y_train)

xgb_prediction = xgboost_clf.predict(X_valid)
xgb_accuracy = accuracy_score(y_valid,xgb_prediction)
print("Training accuracy Score    : ",xgboost_clf.score(X_train,y_train))
print("Validation accuracy Score : ",xgb_accuracy )
print(classification_report(xgb_prediction,y_valid))


# Support vector machine
from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, y_train)

svc_prediction = svc.predict(X_valid)
svc_accuracy = accuracy_score(y_valid,svc_prediction)
print("Training accuracy Score    : ",svc.score(X_train,y_train))
print("Validation accuracy Score : ",svc_accuracy )
print(classification_report(svc_prediction,y_valid))


# Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

logreg_prediction = logreg.predict(X_valid)
logreg_accuracy = accuracy_score(y_valid,logreg_prediction)
print("Training accuracy Score    : ",logreg.score(X_train,y_train))
print("Validation accuracy Score : ",logreg_accuracy )
print(classification_report(logreg_prediction,y_valid))


# Catboost Algorithm
from catboost import CatBoostClassifier, Pool, cv
from sklearn.metrics import accuracy_score
clf2 = CatBoostClassifier()


clf2.fit(X_train, y_train,  
        eval_set=(X_valid, y_valid), 
        verbose=False
)

print('CatBoost model is fitted: ' + str(clf2.is_fitted()))
print('CatBoost model parameters:')
print(clf2.get_params())
catboost_prediction = clf2.predict(X_valid)
catboost_accuracy = accuracy_score(y_valid,catboost_prediction)
print("Training accuracy Score    : ",clf2.score(X_train,y_train))
print("Validation accuracy Score : ",catboost_accuracy )
print(classification_report(catboost_prediction,y_valid))


# #### Evaluation of multiclass classification

# In[18]:


#evaluation
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 
              'Stochastic Gradient Decent', 'XGBoost','CatBoost'],
    'Test accuracy': [svc_accuracy, logreg_accuracy, 
              rf_accuracy, NB_accuracy, 
              sgd_accuracy, xgb_accuracy,catboost_accuracy]})

models.sort_values(by='Test accuracy', ascending=False)


# ### Binary Classification

# In[20]:


cb = train_data[['Tweet','Sentiment']]
cb["Sentiment"]= cb["Sentiment"].replace('Positive',1) 
cb["Sentiment"]= cb["Sentiment"].replace('Extremely Positive',1) 
cb["Sentiment"]= cb["Sentiment"].replace('Neutral',1) 
cb["Sentiment"]= cb["Sentiment"].replace('Negative',0) 
cb["Sentiment"]= cb["Sentiment"].replace('Extremely Negative',0)
X = cb.drop('Sentiment', axis=1)
y = cb.Sentiment
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
cb['Tweet'].apply(lambda x: [item for item in x if item not in stop])
from sklearn.model_selection import train_test_split

train,valid = train_test_split(cb,test_size = 0.2,random_state=0,stratify = cb.Sentiment.values) #stratification means that the train_test_split method returns training and test subsets that have the same proportions of class labels as the input dataset.
print("train shape : ", train.shape)
print("valid shape : ", valid.shape)

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
stop = list(stopwords.words('english'))
vectorizer = CountVectorizer(decode_error = 'replace',stop_words = stop)

X_train = vectorizer.fit_transform(train.Tweet.values)
X_valid = vectorizer.transform(valid.Tweet.values)

y_train = train.Sentiment.values
y_valid = valid.Sentiment.values

print("X_train.shape : ", X_train.shape)
print("X_train.shape : ", X_valid.shape)
print("y_train.shape : ", y_train.shape)
print("y_valid.shape : ", y_valid.shape)


# Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB

naiveByes_clf = MultinomialNB()

naiveByes_clf.fit(X_train,y_train)

NB_prediction = naiveByes_clf.predict(X_valid)
NB_accuracy = accuracy_score(y_valid,NB_prediction)
print("training accuracy Score    : ",naiveByes_clf.score(X_train,y_train))
print("Validation accuracy Score : ",NB_accuracy )
print(classification_report(NB_prediction,y_valid))


# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier()

rf_clf.fit(X_train,y_train)

rf_prediction = rf_clf.predict(X_valid)
rf_accuracy = accuracy_score(y_valid,rf_prediction)
print("Training accuracy Score    : ",rf_clf.score(X_train,y_train))
print("Validation accuracy Score : ",rf_accuracy )
print(classification_report(rf_prediction,y_valid))


# Logistic Regression 
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

logreg_prediction = logreg.predict(X_valid)
logreg_accuracy = accuracy_score(y_valid,logreg_prediction)
print("Training accuracy Score    : ",logreg.score(X_train,y_train))
print("Validation accuracy Score : ",logreg_accuracy )
print(classification_report(logreg_prediction,y_valid))

# Catboost Algorithm
clf2 = CatBoostClassifier()


clf2.fit(X_train, y_train,  
        eval_set=(X_valid, y_valid), 
        verbose=False
)

print('CatBoost model is fitted: ' + str(clf2.is_fitted()))
print('CatBoost model parameters:')
print(clf2.get_params())
catboost_prediction = clf2.predict(X_valid)
catboost_accuracy = accuracy_score(y_valid,catboost_prediction)
print("Training accuracy Score    : ",clf2.score(X_train,y_train))
print("Validation accuracy Score : ",catboost_accuracy )
print(classification_report(catboost_prediction,y_valid))


# Extreme Gradient Boosting 
import xgboost as xgb

xgboost_clf = xgb.XGBClassifier()

xgboost_clf.fit(X_train, y_train)

xgb_prediction = xgboost_clf.predict(X_valid)
xgb_accuracy = accuracy_score(y_valid,xgb_prediction)
print("Training accuracy Score    : ",xgboost_clf.score(X_train,y_train))
print("Validation accuracy Score : ",xgb_accuracy )
print(classification_report(xgb_prediction,y_valid))


# Support vector machine
from sklearn.svm import SVC

svc = SVC()

svc.fit(X_train, y_train)

svc_prediction = svc.predict(X_valid)
svc_accuracy = accuracy_score(y_valid,svc_prediction)
print("Training accuracy Score    : ",svc.score(X_train,y_train))
print("Validation accuracy Score : ",svc_accuracy )
print(classification_report(svc_prediction,y_valid))


# Stochastic Gradient Descent-SGD Classifier
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(loss = 'hinge', penalty = 'l2', random_state=0)

sgd_clf.fit(X_train,y_train)

sgd_prediction = sgd_clf.predict(X_valid)
sgd_accuracy = accuracy_score(y_valid,sgd_prediction)
print("Training accuracy Score    : ",sgd_clf.score(X_train,y_train))
print("Validation accuracy Score : ",sgd_accuracy )
print(classification_report(sgd_prediction,y_valid))


# #### Evaluation of multiclass classification

# In[21]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 
              'Stochastic Gradient Decent', 'XGBoost','CatBoost'],
    'Test accuracy': [svc_accuracy, logreg_accuracy, 
              rf_accuracy, NB_accuracy, 
              sgd_accuracy, xgb_accuracy,catboost_accuracy]})

models.sort_values(by='Test accuracy', ascending=False)


# - We use 7 classification methods to do our sentiment analysis and achieves 61% accuracy which is the best with CatBoost based on multiclass models. To improve our classification accuracy, we decide to use binary models to do the prediction. As a result, binary classification is much better than multiclass classification. Stochastic Gradient Descent (SGD) Classifier got the best performance with test accuracy 87%.

# In[ ]:




