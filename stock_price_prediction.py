# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 17:36:52 2021

@author: rashm
"""
import nltk
import re
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
data = pd.read_csv(r'C:\Users\rashm\OneDrive\Desktop\Stock-Sentiment-Analysis-f84cfd6da77e34d86081e0aaf6afd6d05050e0c7/Data.csv.csv', encoding='ISO-8859-1')
data.head()
data.columns
y = data['Label']
df = data.drop(['Date', "Label"],axis=1) 
lemmetizer = WordNetLemmetizer()
headlines = []
for i in range(4101):
    headlines.append(' '.join(str(x) for x in df.iloc[i,0:25]))
for i in range(len(headlines)):
    words = nltk.word_tokenize(headlines[i])
    for j in range(len(words)):
        word[j] = re.sub('[^a-zA-Z]', ' ', word[j])
        word[j] = word[j].lower()
        word[j] = stemmer.stem(word[j]) if word[j] not in stopwords.words('english')
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(2,2), max_features=2500)
bow = cv.fit_transform(headlines)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(bow, y, test_size=0.3, random_state=101)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200, criterion='entropy')
rfc.fit(X_train, y_train)
prediction = rfc.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
accuracy_score(y_test, prediction)
report = classification_report(y_test,prediction)
y.value_counts()
