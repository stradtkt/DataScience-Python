# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('USA_Housing.csv')
df.head()
df.info()
df.describe()
df.columns
sns.pairplot(df)
sns.distplot(df['Price'])
sns.heatmap(df.corr(), annot=True)
df.columns
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y = df['Price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

print(lm.intercept_)
lm.coef_
X_train.columns

cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])
cdf

from sklearn.datasets import load_boston
boston = load_boston()
boston.keys()
print(boston['DESCR'])
