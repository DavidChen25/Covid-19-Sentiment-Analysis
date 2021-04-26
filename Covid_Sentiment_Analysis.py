#!/usr/bin/env python
# coding: utf-8

# In[4]:


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
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
nltk.download('stopwords')
nltk.download('vader_lexicon')
from matplotlib import pyplot as plt
from matplotlib import ticker
import seaborn as sns
import plotly.express as px
sns.set(style='darkgrid')


# In[5]:


data2=pd.read_csv('corona_0324.csv')
data1=pd.read_csv('corona_0325.csv')
data3=pd.read_csv('corona_0320_0327.csv')
data4=pd.read_csv('corona_0320_0328.csv')
need_columns=['date','username','text']
data5=pd.merge(data1,data2,how='outer')
data6=pd.merge(data5,data3,how='outer')
data=pd.merge(data6,data4,how='outer')
data=data[need_columns]
data.username=data.username.astype('category')
data.username=data.username.cat.codes
data.date=pd.to_datetime(data.date).dt.date
text=data['text']
remove_url=lambda x: re.sub(r'https\S+' , '',str(x))
text_lr = text.apply(remove_url)

#coverting all tweets to lowercase
to_lower=lambda x: x.lower()
text_lr_lc=text_lr.apply(to_lower)
#remove punctutation
remove_puncts=lambda x: x.translate(str.maketrans('','',string.punctuation))
text_lr_lc_np=text_lr_lc.apply(remove_puncts)
#remove stopwords
more_words=['covid','#coronavirus','coronavirus','coronalockdown','coronavirusoutbreak','covid19']
stop_words=set(stopwords.words('English'))
stop_words.update(more_words)
remove_words=lambda x:' '.join([word for word in x.split() if word not in stop_words])
text_lr_lc_np_ns=text_lr_lc_np.apply(remove_words)
words_list=[word for line in text_lr_lc_np_ns for word in line.split()]
words_list[:5]
word_counts=Counter(words_list).most_common(50)
words_df=pd.DataFrame(word_counts)
words_df.columns=['word','frq']
fig = px.bar(words_df,x='word',y='frq',title='Most common words')
fig.update_xaxes(tickangle = -45)
fig.show()


# In[8]:


data.text=text_lr_lc_np_ns
sid=SentimentIntensityAnalyzer()
ps=lambda x: sid.polarity_scores(x)
sentiment_scores=data.text.apply(ps)
sentiment_df=pd.DataFrame(data=list(sentiment_scores))
labelize=lambda x :'neutral'  if x==0 else('positive' if x>0 else 'negative')
sentiment_df['label']=sentiment_df.compound.apply(labelize)
df=data.join(sentiment_df.label)
count_df=df.label.value_counts().reset_index()
sns.barplot(x='index',y='label',data=count_df)
data_agg=df[['username','date','label']].groupby(['date','label']).count().reset_index()
data_agg.columns=['date','label','counts']
for i in data_agg.index:
    if data_agg['counts'][i]<1000:
        data_agg['counts'][i]=data_agg['counts'][i]*10
    
px.line(data_agg,x='date',y='counts',color='label',title='daily tweets sentimental analysis')


# In[9]:


data


# In[4]:


df_new=pd.DataFrame()
df_new['tweet']=df['text']
df_new['sentiment']=df['label']
df_new['sentiment']=df_new['sentiment'].replace('positive',1)
df_new['sentiment']=df_new['sentiment'].replace('neutral',1)
df_new['sentiment']=df_new['sentiment'].replace('negative',0)
X=df_new.drop('sentiment',axis=1)
y=df_new.sentiment

from sklearn.model_selection import train_test_split

train,test = train_test_split(df_new,test_size = 0.2,random_state=0,stratify = df_new.sentiment.values)
print("train shape : ", train.shape)
print("test shape : ", test.shape)


# In[5]:


from sklearn.feature_extraction.text import CountVectorizer
stop = list(stopwords.words('english'))
vectorizer = CountVectorizer(decode_error = 'replace',stop_words = stop)

X_train = vectorizer.fit_transform(train.tweet.values)
X_test = vectorizer.transform(test.tweet.values)

y_train = train.sentiment.values
y_test = test.sentiment.values

print("X_train.shape : ", X_train.shape)
print("X_test.shape : ", X_test.shape)
print("y_train.shape : ", y_train.shape)
print("y_test.shape : ", y_test.shape)


# ### Naive Bayes Classifier (Binary Classification)

# In[6]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
naiveByes_clf = MultinomialNB()

naiveByes_clf.fit(X_train,y_train)

NB_prediction = naiveByes_clf.predict(X_test)
NB_accuracy = accuracy_score(y_test,NB_prediction)
print("training accuracy Score    : ",naiveByes_clf.score(X_train,y_train))
print("Validation accuracy Score : ",NB_accuracy )
print(classification_report(NB_prediction,y_test))


# ### Random Forest Classifier (Binary Classification)

# In[7]:


from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier()

rf_clf.fit(X_train,y_train)

rf_prediction = rf_clf.predict(X_test)
rf_accuracy = accuracy_score(y_test,rf_prediction)
print("Training accuracy Score    : ",rf_clf.score(X_train,y_train))
print("Validation accuracy Score : ",rf_accuracy )
print(classification_report(rf_prediction,y_test))


# ### Logistic Regression (Binary Classification)

# In[8]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

logreg_prediction = logreg.predict(X_test)
logreg_accuracy = accuracy_score(y_test,logreg_prediction)
print("Training accuracy Score    : ",logreg.score(X_train,y_train))
print("Validation accuracy Score : ",logreg_accuracy )
print(classification_report(logreg_prediction,y_test))


# ### Support vector machine (Binary Classification)

# In[9]:


from sklearn.svm import SVC

svc = SVC()

svc.fit(X_train, y_train)

svc_prediction = svc.predict(X_test)
svc_accuracy = accuracy_score(y_test,svc_prediction)
print("Training accuracy Score    : ",svc.score(X_train,y_train))
print("Validation accuracy Score : ",svc_accuracy )
print(classification_report(svc_prediction,y_test))


# ### Extreme Gradient Boosting (Binary Classification)

# In[10]:


import xgboost as xgb

xgboost_clf = xgb.XGBClassifier()

xgboost_clf.fit(X_train, y_train)

xgb_prediction = xgboost_clf.predict(X_test)
xgb_accuracy = accuracy_score(y_test,xgb_prediction)
print("Training accuracy Score    : ",xgboost_clf.score(X_train,y_train))
print("Validation accuracy Score : ",xgb_accuracy )
print(classification_report(xgb_prediction,y_test))


# ### Catboost Algorithm (Binary Classification)

# In[11]:


from catboost import CatBoostClassifier, Pool, cv
from sklearn.metrics import accuracy_score

clf2 = CatBoostClassifier()


clf2.fit(X_train, y_train,  
        eval_set=(X_test, y_test), 
        verbose=False
)

print('CatBoost model is fitted: ' + str(clf2.is_fitted()))
print('CatBoost model parameters:')
print(clf2.get_params())


# In[12]:


catboost_prediction = clf2.predict(X_test)
catboost_accuracy = accuracy_score(y_test,catboost_prediction)
print("Training accuracy Score    : ",clf2.score(X_train,y_train))
print("Validation accuracy Score : ",catboost_accuracy )
print(classification_report(catboost_prediction,y_test))


# ### Stochastic Gradient Descent-SGD Classifier (Binary Classification)

# In[13]:


from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(loss = 'hinge', penalty = 'l2', random_state=0)

sgd_clf.fit(X_train,y_train)

sgd_prediction = sgd_clf.predict(X_test)
sgd_accuracy = accuracy_score(y_test,sgd_prediction)
print("Training accuracy Score    : ",sgd_clf.score(X_train,y_train))
print("Validation accuracy Score : ",sgd_accuracy )
print(classification_report(sgd_prediction,y_test))


# # Winner Model:

# ## Stochastic Gradient Descent-SGD Classifier (Binary Classification)

# In[14]:


# Get the predicted classes
train_class_preds = sgd_clf.predict(X_train)
test_class_preds = sgd_clf.predict(X_test)


# In[15]:


from sklearn.metrics import mean_squared_error,mean_absolute_error, make_scorer,classification_report,confusion_matrix,accuracy_score,roc_auc_score,roc_curve

# Get the confusion matrix for both train and test. 

labels = ['Negative', 'Positive']
cm = confusion_matrix(y_train, train_class_preds)
print(cm)

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax) #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)


# In[16]:


# Let's check the overall accuracy. Overall accuracy is very good.
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

y_pred = sgd_clf.predict(X_test)

score =accuracy_score(y_test,y_pred)
print('The accuracy is', score)


# In[17]:


# F1 score for our classifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


y_pred =  sgd_clf.predict(X_test)
print(f1_score(y_test,y_pred, average="macro"))


# In[18]:


#score is mean accuracy
scikit_score = sgd_clf.score(X_test,y_test)
print('scikit score:', scikit_score)


# In[19]:


# Recall score for our winner model
recall_score(y_test, y_pred, average='macro')


# In[21]:


# Classification Report for our stochastic gradient descent algorithm
classification_report(y_test,y_pred)


# In[22]:


# Very low type 1 and type 2 error
confusion_matrix(y_test,y_pred)


# # Evaluation of all Binary Classification Models
# #### All the model test accuracy by descending order

# In[23]:


models = pd.DataFrame({
    'Model': ['Naive Bayes', 'Random Forest', 'Logistic Regression', 'Support Vector Machines',  
              'Extreme Gradient Boost', 'CatBoost', 'Stochastic Gradient Decent'],
    'Test accuracy': [NB_accuracy, rf_accuracy, logreg_accuracy, svc_accuracy,   
               xgb_accuracy, catboost_accuracy, sgd_accuracy]})

models.sort_values(by='Test accuracy', ascending=False)


# In[ ]:




