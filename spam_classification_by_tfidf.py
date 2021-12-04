# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 12:04:55 2021

@author: rashm
"""
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
messages = pd.read_csv(r'C:\Users\rashm\OneDrive\Desktop\NLP\SMSSpamCollection.csv', sep='\t', names=['label','message'])
#cleaning of data
import re
corpus=[]
lemmatizer = WordNetLemmatizer()
for i in range(len(messages['message'])):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
#creating model with tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
spam_classi_model = TfidfVectorizer(max_features=5000)
X = spam_classi_model.fit_transform(corpus).toarray()
y = messages['label']
y = pd.get_dummies(y, drop_first=True)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=101)
from sklearn.naive_bayes import MultinomialNB
spam_model = MultinomialNB().fit(X_train,y_train)
pred1= spam_model.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(y_test, pred1)
accuracy_score(y_test, pred1)
