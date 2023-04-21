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

def vectorize(data):
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
    return vectors, labels


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def log_train(data, labels, learning_rate, num_iter):
    # Initialize the parameters
    num_features = data.shape[1]
    w = np.zeros((num_features, 1))
    b = 0

    # Define the cost function
    m = data.shape[0]

    def cost_function():
        z = np.dot(data, w) + b
        predict = sigmoid(z)
        J = (-1 / m) * np.sum(labels * np.log(predict) + (1 - labels) * np.log(1 - predict))
        return J

    # Implement gradient descent
    for i in range(num_iter):
        z = np.dot(data, w) + b
        predict = sigmoid(z)
        w -= learning_rate * (1 / m) * np.dot(data.T, predict - labels)
        b -= learning_rate * (1 / m) * np.sum(predict - labels)

        # Print the cost every 100 iterations
        if i % 100 == 0:
            cost = cost_function()
            print(f"Iteration {i}: Cost = {cost}")

    # Return the trained parameters
    return w, b


def log_predict(data, weights):
    """Make predictions using a logistic regression model"""
    z = np.dot(data, weights)
    return sigmoid(z)


def test_log_accuracy(data, labels, freqs, weights):
    pred = []
    for tweet in data:
        label = log_predict(tweet, freqs, weights)
        if label > 0.5:
            pred.append(1)
        else:
            pred.append(0)
    pred = np.array(pred)
    test_y = labels.reshape(-1)
    accuracy = np.sum((test_y == pred).astype(int)) / len(data)

    return accuracy