"""
import pandas
import nltk
nltk.download('punkt')
import string
import re
import contractions
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
def train(data):
    # Array of tweets and labels
    tweets = np.array(data['tweet'])
    labels = np.array(data['label'])

    # Number of total positive and negative labels
    positives = np.sum(labels == 1)
    negatives = np.sum(labels == 0)

    # Prior probabilities
    pos_prior = positives / len(labels)
    neg_prior = negatives / len(labels)

    # Turn the list of tweets into a single list of words
    tweets_words = []
    for tweet1 in tweets:
        for word in tweet1:
            tweets_words.append(word)

    # Set of unique words
    vocabulary = set(tweets_words)

    # Count of positive and negative labels for each word
    pos_word_count = np.zeros(len(vocabulary))
    neg_word_count = np.zeros(len(vocabulary))
    """"""
        for i, word in enumerate(vocabulary):
            # For each tweet in the list of tweets
            for j in range(len(tweets)):
                # If word is in the tweet, increment positive or negative count of the word
                if word in tweets[j]:
                    if labels[j] == 1:
                        pos_word_count[i] += 1
                    elif labels[j] == 0:
                        neg_word_count[i] += 1
    """
"""
    print(len(vocabulary))
    for i, word in enumerate(vocabulary):
        print(i)
        pos_word_count[i] = np.sum([word in tweet2 and label == 1 for tweet2, label in zip(tweets, labels)])
        neg_word_count[i] = np.sum([word in tweet2 and label == 0 for tweet2, label in zip(tweets, labels)])

    log_pos_prior = np.log(pos_prior)
    log_neg_prior = np.log(neg_prior)

    # Array of label log likelihood for each word
    log_likelihood_positive = {}
    log_likelihood_negative = {}
    for i, word in enumerate(vocabulary):
        print(i)
        log_likelihood_positive[word] = np.log((pos_word_count[i] + 1) / (positives + len(vocabulary)))
        log_likelihood_negative[word] = np.log((neg_word_count[i] + 1) / (negatives + len(vocabulary)))

    return log_pos_prior, log_neg_prior, log_likelihood_positive, log_likelihood_negative


def predict(tweets, log_pos_prior, log_neg_prior, log_like_pos, log_like_neg):
    log_prob_positive = log_pos_prior
    log_prob_negative = log_neg_prior
    for word in tweets:
        print(word)
        if word in log_like_pos:
            log_prob_positive += log_like_pos[word]
        if word in log_like_neg:
            log_prob_negative += log_like_neg[word]
    # Use the log sum exp trick to avoid numerical underflow
    prob_positive = np.exp(log_prob_positive - np.logaddexp(log_prob_positive, log_prob_negative))
    if prob_positive > 0.5:
        return 1
    else:
        return 0


data1 = process(train_data)
print(data1.head())
log_pos, log_neg, log_like_positive, log_like_negative = train(data1)
tweet = "half way through the website now and #allgoingwell very"
sentiment = predict(tweet, log_pos, log_neg, log_like_positive, log_like_negative)
print(sentiment)
"""