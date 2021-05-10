# Sentiment Analysis of Covid-19 Tweets
#### Chen Yuqi & Zhang Guoran


# Abstract
  COVID-19 pandemic happens on 11th March 2020 and it rapidly developed as one of the most wildly pandemic in human history. This study focuses on the sentiment analysis of tweets of the Twitter social media using Python programming language. The dataset has been collected from Twitter that contains 41157 tweets. The primary object is to use a sentiment classifier to study people’s attitudes to corona virus and predict. This report uses 7 different classification methods to predict the tweets contents and the corresponding attitudes. Results indicate that Catboost is the best model for multiclass classification with 61% accuracy and SGD is the best model for binary classification with 87% accuracy.
- _Keywords— —COVID-19; pandemic; sentiment analysis; Twitter; Catboost; SGD_

# Introduction
COVID-19, Corona Virus Disease of 2019, has been stated as a pandemic by World Health Organization (WHO) on 11th March 2020. The current situation has reported that there are still more than twenty-two million people being tested positive worldwide as of 9th May 2021 [1]. People still need work remoted in many countries, self-quarantine, and keep social distance from each other to control this virus. Thus, it is critical to implement different measures to protect countries by elucidating relevant facts and information [1]. This pandemic is currently causing severe economic problems on both a national and microscale. The nations may experience economic recession since people have being restricted from traveling, and all economic activities have been closed [2]. As the end of 2020, many vaccines have been coming out through many countries especially the China and the United States of America. Many people are getting recovered after taking those vaccines in those two countries, but there is still a large amount of people in the U.S. not taking vaccines for different reasons.  

The amount of social data on the internet is rapidly increasing these days. This allows researchers to gain access to data and information for research purposes [3]. During the global COVID-19 outbreak, many individuals and organizations have posted their thoughts on the coronavirus Twitter has been using as one of social networking sites that allows people to share information and comments during this crucial time. Many people, organizations, and even governments rely on this media to get news or post their thoughts and feelings about the issues of the pandemic.  

Tweets are the messages that created by using Twitter. These data are freely available in the public domain [2]. Then many people can access and collect those tweets data from Twitter by using some Twitter API, but it has limits that change over time and only allows one week’s data per time. Some companies using these data to review some customers feelings and thoughts which can help them to change some of their decisions or policies, and this type of study is often called the sentiment analysis  

With the help of natural language processing (NLP) techniques, we can create more complex discussion by using sentiment analysis, and word cloud visualizations. This study focuses on the sentiment analysis of tweets of the Twitter social media using Python programming language. The tweets have been collected, and it will be preprocessed and then used for text manipulation and sentiment analysis by using some packages such like Nltk. After the sentiment analysis, we will build a classification model to predict the sentiment of Covid-19 tweets.  

# Related Work 
Ayush Pareek uses different feature sets like unigrams, bigrams, trigrams and negation detection and machine learning classifiers to find the best combination for sentiment analysis of twitter. He also manipulates different pre-processing steps like - punctuations, emoticons, twitter specific terms and stemming methods. He trains classifier by using some different machine-learning algorithms like the Naive Bayes, Decision Trees and Maximum Entropy. He concludes that both Negation Detection and higher order n-grams are useful for the text classification and Naive Bayes Classifier performs better than Maximum Entropy Classifier. He gets the best accuracy of 86.68% in the case of Unigrams + Bigrams + Trigrams combination which is trained on the Naive Bayes Classifier [4].  

Our vectorization methods coding (N-grams & Negation handling) are from [Ayush Pareek Github](https://github.com/ayushoriginal/Sentiment-Analysis-Twitter):   

# Method
## About the data
We collect our data from [Kaggle](https://www.kaggle.com/datatattle/covid-19-nlp-text-classification?select=Corona_NLP_train.csv) about the Coronavirus tweets. The tweets have been pulled from Twitter and manual tagging has been done then. For this project, more than 40 thousand of tweets with the "COVID-19" related keywords between March 2020 to April 2020 were fetched for the sentiment analysis. The data originally contains 6 variables – UserName, ScreenName, Location, TweetAt, OriginalTweet, and Sentiment. The ScreeNames and UserNames have been given codes to avoid any privacy concerns. The data has already split into training and testing datasets with 41157 tweets in training dataset and 3798 tweets in testing dataset. The Sentiment has been transformed into 5 types of sentiments – extremely negative, negative, neutral, positive, extremely positive. TweetAt is the time variable which indicates the time the tweets have been posted. The originalTweet is the tweet texts we want to use to perform our sentiment analysis. So in next section we will introduce our data preprocessing steps in details.   
  
## Data Preprocessing

Remove those columns that are not useful for our sentiment analysis and only keep ‘OriginalTweet and ‘Sentiment’ in the data frame. Then we remove those emoticons, urls, userhandels, stopwords, negations, and special characters.  

We use TextBlob package to visualize the sentiments with its polarity and subjectivity and count figure.  

![1](https://user-images.githubusercontent.com/54686263/117685734-305af980-b184-11eb-82bf-a7e3a2f97dbd.png) 

Most of the tweets in the dataset seem to be neutral and not much subjectivity. However, in the polar tweets, there are slightly more positively charged tweets than negatively charged tweets. We can say that covid-19 pandemic on twitter is generally optimistic, but it would be nice to see what sentiment that people’s feelings and thoughts are [5].  

![image](https://user-images.githubusercontent.com/54686263/117685977-6a2c0000-b184-11eb-8cdc-8a90aea44143.png)  

The original common words showing above include those high frequent hashtags like “COVID”, “covid-19”, and “CORONAVIRUS” etc. We tend to remove these words and only keep “coronavirus” in the list. There are some high frequent words such like “price”, “food”, “consumer”, and “pandemic” etc.  

Then we want to know that what is the common word difference between the “Positive” and “Negative” sentiment. We introduce the word-cloud map to show the comparison as figure 4 and figure 5 below. These word clouds yield similar result as the common words in figure 3. The extremely positive sentiment shows the reference words like “great”, “best”, “thank”, “hand-sanitizer”, and “free”. Moreover, the extremely negative sentiment shows the reference words like “panic”, “crisis”, and “fear” So I think it is more likely want to emphasis lots of people are experiencing the crisis period and feel panic and fear of the virus, but those people who get over the virus and get recovered are more optimal like they feel more thankful to the doctors or those vaccines that helps them recover.  

![image](https://user-images.githubusercontent.com/54686263/117686300-bb3bf400-b184-11eb-977f-0efbb4f3b543.png)  
Figure 4: WordCloud of Extremly Positive Sentiment  

![image](https://user-images.githubusercontent.com/54686263/117686316-becf7b00-b184-11eb-8ea2-4cd9d3bf9303.png)  
Figure 5: WordCloud of Extremly Negative Sentiment  

## Vectorization  

1. Tf-IDF: It stands for “Term Frequency — Inverse Document Frequency”. This is a common technique to quantify a word in documents, it generally computes a weight to each word which signifies the importance of the word in the document and corpus [6]. By calculating Tf-IDF, we can easily see the most frequent words appears in the tweet texts. Then it filters those commonly words and remains the important words. We are using sklearn feature extraction package to vectorizing the tweets with tf-idf.  
2. Part-of-speech taggings: A POS tag (or part-of-speech tag) is a special label assigned to each word in a text corpus to indicate the part of speech like “N” stands for noun and “A” stands for adjective etc [7]. With this tool, we can generate and remove those common tagsets to keep remaining important features.  
3. CountVectorizer: It is a common feature numerical calculation class, which is a text feature extraction method. For each training text, it only considers the frequency of each word in the training text. We use CounterVectorizer function in sklearn package to do our analysis. The CountVectorizer converts words in the text into a word frequency matrix, which counts the number of occurrences of each word using the FIT_Transform function. There are many parameters in CountVectorizer classes.It has 3 procedures which are preprocessing、tokenizing and n-grams generation. Word matrix elements a[i][j] represent the word frequency of j words in the ith text. That is, the number of occurrences of each word, get_feature_names() to see the keywords for all the text, and toarray() to see the results of the word frequency matrix.  

# Classification  

We split the whole dataset into training dataset with 80% and testing dataset with 20% of the whole dataset.  

We set the stopwords as ‘English’ and add our stopwords list on it. Then using CountVectorizer function on tweet text we got in training and testing dataset and transform them into word matrix. Data shape after vecotrizer are shown below:  

![2](https://user-images.githubusercontent.com/54686263/117687666-01458780-b186-11eb-8a14-12318d32d980.png)

For our research, we used 7 different classification models to do the sentiment analysis which are Naive Bayes, Logistic Regression, Random Forest, Support Vector Machines, CatBoost, XGboost, Stochastic Gradient Decent.  

# Results  
## Results for multiclass classification
- Our winning model: Catboost classifier  

![cat](https://user-images.githubusercontent.com/54686263/117687898-436ec900-b186-11eb-8598-37c267b75aab.png) 

## Results for binary classification  
- Our Winner model: Stochastic Gradient Decent classifier  

![SGD](https://user-images.githubusercontent.com/54686263/117688007-61d4c480-b186-11eb-95e5-986610584fce.png)  

## Evaluating all classification models  
- For multiClass classification models:  

![mul](https://user-images.githubusercontent.com/54686263/117688620-f0e1dc80-b186-11eb-8e96-e8298769b8da.png)  
- For Binary Classification models:  

![bi](https://user-images.githubusercontent.com/54686263/117688670-fccd9e80-b186-11eb-8621-011d5c0c458a.png)  

# Conclution  
We create a sentiment classifier for twitter using labelled data sets. We also investigate the relevance of using multiclass classifier and binary classifier for the purpose of sentiment analysis. To test our classification results, we split the dataset into train group with 80% of the whole dataset and test group with 20% of the whole dataset.  

We use 7 classification methods to do our sentiment analysis and achieves 61% accuracy which is the best with CatBoost based on multiclass models. To improve our classification accuracy, we decide to use binary models to do the prediction. As a result, binary classification is much better than multiclass classification. Stochastic Gradient Descent (SGD) Classifier got the best performance with test accuracy 87%.  

# Limitations  
The sentiments are not labeled precisely since there are still existing some ambiguous sentences or sarcasms in the tweet’s sentiment. So the result seems to be more general but not show in detailed or precise.  

# Future Work  

We could do more classification models or topic modeling in the future for different topic.  

The other topic such as vaccines could also help us to explore what are people saying with respect to side effects

# References 

[1] K. Chakraborty, S. Bhatia, S. Bhattacharyya, J. Platos, R. Bag, and A. E. Hassanien, “Sentiment Analysis of COVID-19 tweets by Deep Learning Classifiers-A study to show how popularity is affecting accuracy in social media,” Applied Soft Computing, 28-Sep-2020. [Online]. Available: https://www.sciencedirect.com/science/article/abs/pii/S156849462030692X. [Accessed: 09-May-2021].  

[2] M. A. Kausar, A. Soosaimanickam, and M. Nasar, “Public Sentiment Analysis on Twitter Data during COVID-19 Outbreak,” International Journal of Advanced Computer Science and Applications, vol. 12, no. 2, 2021.  

[3] B. P. Pokharel, “Twitter Sentiment Analysis During Covid-19 Outbreak in Nepal,” SSRN, 15-Jun-2020. [Online]. Available: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3624719. [Accessed: 09-May-2021].  

[4] A. Pareek, “ayushoriginal/Sentiment-Analysis-Twitter,” GitHub, 20-Oct-2016. [Online]. Available: https://github.com/ayushoriginal/Sentiment-Analysis-Twitter. [Accessed: 10-May-2021].   

[5] S. Dua, “Sentiment Analysis of COVID-19 Vaccine Tweets,” Medium, 29-Mar-2021. [Online]. Available: https://towardsdatascience.com/sentiment-analysis-of-covid-19-vaccine-tweets-dc6f41a5e1af. [Accessed: 10-May-2021].  

[6] W. Scott, “TF-IDF for Document Ranking from scratch in python on real world dataset.,” Medium, 21-May-2019. [Online]. Available: https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089. [Accessed: 10-May-2021].  

[7] “POS tags and part-of-speech tagging,” Sketch Engine, 25-Sep-2020. [Online]. Available: https://www.sketchengine.eu/blog/pos-tags/. [Accessed: 10-May-2021].  
