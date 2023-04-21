import pandas
import nltk
import string
import re
import contractions
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


train_data = pandas.read_csv('../twitter_Sentiment_ML_Project/archive/train.csv')
test_data = pandas.read_csv('../twitter_Sentiment_ML_Project/archive/test.csv')


def process(data):
    # Removes @users
    data['tweet'] = data['tweet'].apply(lambda x: re.sub(r'@\w+', '', x))

    # Expand contraction
    data['tweet'] = data['tweet'].apply(lambda x: contractions.fix(x))

    # Removes characters that are not from the english alphabet
    data['tweet'] = data['tweet'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

    # Removes letters that occur more than two times in a row
    data['tweet'] = data['tweet'].apply(lambda x: re.sub(r'(.)\1{2,}', r'\1\1', x))

    # Removes punctuations
    # Note: right now it gets rid of apostrophes
    data['tweet'] = data['tweet'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

    # Tokenizes each word
    data['tweet'] = data['tweet'].apply(lambda x: word_tokenize(x))

    # Removes stop words
    stop_words = set(stopwords.words('english'))
    data['tweet'] = data['tweet'].apply(lambda x: list(filter(lambda word: word.casefold() not in stop_words, x)))

    # Convert the words to lowercase
    data['tweet'] = data['tweet'].apply(lambda x: [word.lower() for word in x])

    # Reduce words to root
    data['tweet'] = data['tweet'].apply(lambda x: [PorterStemmer().stem(word) for word in x])

    return data


def feature_extract(data):
    # array of tweets and labels
    tweets = np.array(data['tweet'])
    labels = np.array(data['label'])

    # Turn list of tweets into single list of words
    tweet_words = []
    for tweet in tweets:
        for word in tweet:
            tweet_words.append(word)
    # Set of unique words
    vocabulary = set(tweet_words)

    vectors = np.zeros((len(tweets), len(vocabulary)))
    for i, tweet in enumerate(tweets):
        for j, word in enumerate(vocabulary):
            vectors[i, j] = tweet.count(word)
    return vectors, labels, vocabulary


def log_train(data, labels, learning_rate, num_iter):
    # Initialize the parameters
    new_data, labels, vocabulary = feature_extract(data)
    num_features = len(vocabulary)
    w = np.zeros(num_features)
    b = 0

    # Define the cost function
    m = new_data.shape[0]


    # Implement gradient descent
    for i in range(num_iter):
        z = np.dot(new_data, w) + b
        predict = 1 / (1 + np.exp(-z))
        w -= learning_rate * (1 / m) * np.dot(new_data.T, predict - labels)
        b -= learning_rate * (1 / m) * np.sum(predict - labels)
        J = (-1 / m) * np.sum(labels * np.log(predict) + (1 - labels) * np.log(1 - predict))

        # Print the cost every 100 iterations
        if i % 100 == 0:
            cost = J
            print(f"Iteration {i}: Cost = {cost}")

    # Return the trained parameters
    return J, w, b


def log_predict(data, weights, b):
    """Make predictions using a logistic regression model"""
    z = np.dot(data, weights) + b
    return 1 / (1 + np.exp(-z))


def test_log_accuracy(data, labels, weights, b):
    predict = []
    new_data, new_l, vocab = feature_extract(data)
    new_labels = log_predict(new_data, weights, b)
    for label in new_labels:
        if label > 0.5:
            predict.append(1)
        else:
            predict.append(0)
    predict = np.array(predict)
    acc = (predict == np.squeeze(labels)).sum() / len(data)

    return acc


J, w, b = log_train(process(train_data), train_data['label'], 0.01, 2000)
accuracy = test_log_accuracy(process(test_data), test_data['label'], w, b)
print(accuracy)
