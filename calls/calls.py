# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('911.csv')

df.info

df.head()

df['zip'].value_counts()

df['twp'].value_counts().head()

x = df['title'].iloc[0]
x.split(':')[0]
df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])
df['Reason'].head()

df['Reason'].value_counts()

sns.countplot(x='Reason', data=df, palette='viridis')

df.info()

type(df['timeStamp'].iloc[0])

df['timeStamp'] = pd.to_datetime(df['timeStamp'])
df['timeStamp']

time = df['timeStamp'].iloc[0]
time.hour
time.year

df['hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)
df.head()

dmap = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
df['Day of Week'] = df['Day of Week'].map(dmap)
df.head()

sns.countplot(x='Day of Week', data=df, hue='Reason', palette='viridis')

byMonth = df.groupby('Month').count()
byMonth.head()

byMonth['lat'].plot()

sns.lmplot(x='Month', y='twp', data=byMonth.reset_index())

t = df['timeStamp'].iloc[0]
t

t.date()

df['Date'] = df['timeStamp'].apply(lambda t: t.date())
df.head()

df.groupby('Date').count().head()

df.groupby('Date').count()['lat'].plot()
plt.tight_layout()

df[df['Reason']=='Traffic'].groupby('Date').count()['lat'].plot()
plt.tight_layout()

df.groupby(by=['Day of Week', 'hour']).count()

dayHour = df.groupby(by=['Day of Week', 'hour']).count()['Reason'].unstack()
dayHour

plt.figure(figsize=(12,6))
sns.heatmap(dayHour, cmap='viridis')

sns.clustermap(dayHour, cmap='viridis')

dayMonth = df.groupby(by=['Day of Week', 'Month']).count()['Reason'].unstack()
dayMonth.head()

plt.figure(figsize=(12,6))
sns.heatmap(dayMonth, cmap='viridis')

sns.clustermap(dayMonth, cmap='viridis')





















