import pandas
import nltk
import string
import re
import contractions
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


# Load the data from a CSV file
train_data = pandas.read_csv('../pythonProject/archive/train.csv')

# Removes @users
train_data['tweet'] = train_data['tweet'].apply(lambda x: re.sub(r'@\w+', '', x))

# Expand contraction
train_data['tweet'] = train_data['tweet'].apply(lambda x: contractions.fix(x))

# Removes characters that are not from the english alphabet
train_data['tweet'] = train_data['tweet'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

# Removes letters that occur more than two times in a row
train_data['tweet'] = train_data['tweet'].apply(lambda x: re.sub(r'(.)\1{2,}', r'\1\1', x))

# Removes punctuations
# Note: right now it gets rid of apostrophes
train_data['tweet'] = train_data['tweet'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

# Tokenizes each word
train_data['tweet'] = train_data['tweet'].apply(lambda x: word_tokenize(x))

# Removes stop words
stop_words = set(stopwords.words('english'))
train_data['tweet'] = train_data['tweet'].apply(lambda x: list(filter(lambda word: word.casefold() not in stop_words, x)))

# Convert the words to lowercase
train_data['tweet'] = train_data['tweet'].apply(lambda x: [word.lower() for word in x])

# Reduce words to root
train_data['tweet'] = train_data['tweet'].apply(lambda x: [PorterStemmer().stem(word) for word in x])

tweets = train_data['tweet']
labels = train_data['label']



