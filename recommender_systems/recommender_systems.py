# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=column_names)
df.head()

movie_titles = pd.read_csv('Movie_Id_Titles')
movie_titles.head()

df = pd.merge(df, movie_titles, on='item_id')
df.head()

sns.set_style('white')
df.groupby('title')['rating'].mean().sort_values(ascending=False).head()

df.groupby('title')['rating'].count().sort_values(ascending=False).head()

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()

ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
ratings.head()
ratings['num of ratings'].hist(bins=70)

ratings['rating'].hist(bins=70)

sns.jointplot(x='rating', y='num of ratings', data=ratings, alpha=0.5)