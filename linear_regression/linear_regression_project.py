# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


customers = pd.read_csv('Ecommerce Customers')
customers.head()
customers.describe()

sns.jointplot(data=customers, x='Time on Website', y='Yearly Amount Spent')

sns.jointplot(data=customers, x='Time on App', y='Yearly Amount Spent')

sns.jointplot(x='Time on App', y='Length of Membership', kind='hex', data=customers)

sns.pairplot(customers)

sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=customers)

customers.columns

y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
lm.coef_

predictions = lm.predict(X_test)
plt.scatter(y_test, predictions)
plt.xlabel('Y test (True values)')
plt.ylabel('Predicted Values')

from sklearn import metrics
print('MAE', metrics.mean_absolute_error(y_test, predictions))
print('MSE', metrics.mean_squared_error(y_test, predictions))
print('RMSE', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

metrics.explained_variance_score(y_test, predictions)

sns.distplot((y_test-predictions), bins=50)

cdf = pd.DataFrame(lm.coef_, X.columns, columns=['Coeff'])
cdf

