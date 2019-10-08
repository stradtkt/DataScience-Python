# -*- coding: utf-8 -*-

import nltk
# nltk.download_shell()
messages = [line.rstrip() for line in open('SMSSpamCollection')]
print(len(messages))
messages[0]
for mess_no, message in enumerate(messages[:10]):
    print(mess_no, message)
    print('\n')
    
import pandas as pd
messages = pd.read_csv('SMSSpamCollection', sep='\t', names=['label','message'])
messages.head()

messages.describe()

messages.groupby('label').describe()

messages['length'] = messages['message'].apply(len)
messages.head()

import matplotlib.pyplot as plt
import seaborn as sns
messages['length'].plot.hist(bins=50)

messages[messages['length'] == 910]

messages[messages['length'] == 910]['message'].iloc[0]

messages.hist(column='length', by='label', bins=60, figsize=(12,4))
import string
mess = 'Sample message! Notice: it has punctuation.'
nopunc = [c for c in mess if c not in string.punctuation]
nopunc

from nltk.corpus import stopwords
stopwords.words('english')

nopunc = ''.join(nopunc)
nopunc

X = ['a','b','c','d']
''.join(X)

clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
clean_mess

def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

messages.head()

messages['message'].head().apply(text_process)

from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process)
bow_transformer.fit(messages['message'])

print(len(bow_transformer.vocabulary_))

mess4 = messages['message'][3]
print(mess4)

bow4 = bow_transformer.transform([mess4])
print(bow4)

print(bow4.shape)

bow_transformer.get_feature_names()[4068]

















