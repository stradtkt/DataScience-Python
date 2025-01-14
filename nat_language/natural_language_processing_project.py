# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

yelp = pd.read_csv('yelp.csv')
yelp.head()
yelp.info()
yelp.describe()

yelp['text length'] = yelp['text'].apply(len)

sns.set_style('white')
g = sns.FacetGrid(yelp, col='stars')
g.map(plt.hist, 'text length', bins=50)

sns.boxplot(x='stars', y='text length', data=yelp, palette='rainbow')

sns.countplot(x='stars', data=yelp, palette='rainbow')

stars = yelp.groupby('stars').mean()
stars
stars.corr()

sns.heatmap(stars.corr(), cmap='coolwarm', annot=True)

yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]
yelp_class.info()

X = yelp_class['text']
y = yelp_class['stars']

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)

predictions = nb.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('model', MultinomialNB())        
])

X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

pipe.fit(X_train, y_train)

predictions = pipe.predict(X_test)
print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))






