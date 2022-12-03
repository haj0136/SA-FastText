import re
import pandas as pd
import numpy as np
import nltk
from skift import FirstColFtClassifier
from sklearn.model_selection import cross_val_score
from nltk.corpus import stopwords
import time

nltk.download('stopwords')


def show_score(_score):
    print(_score)
    print(f"Average Score: {np.mean(_score)}")


resultsColumns = ["index", "lossFunction", "learningRate", "NOiter", "accuracy", "time"]

# EXPERIMENT 3

# # Amazon
# train data
data_train = pd.read_csv('../amazon/AmazonTrainSet1M.tsv', sep='\t', header=0, encoding="utf-8")
data_train['SentimentText'] = data_train['SentimentText'].str.lower()

# test data
data_test = pd.read_csv('../amazon/AmazonTestSet400k2.tsv', sep='\t', header=0, encoding="utf-8")
data_test['SentimentText'] = data_test['SentimentText'].str.lower()

stop_words = set(stopwords.words("english"))


def remove_stop_words(text):
    text = [word for word in text.split() if not word in stop_words]
    text = " ".join(text)
    return text


def remove_punctuation(text):
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    return text


def preprocess_train(df_train, df_test, functions, word_ngrams=1, _iterations=5, _lr=0.1, _loss="softmax"):
    _data_train = pd.DataFrame(df_train['SentimentText'])
    _data_test = pd.DataFrame(df_test['SentimentText'])
    for function in functions:
        _data_train['SentimentText'] = _data_train['SentimentText'].apply(lambda x: function(x))
        _data_test['SentimentText'] = _data_test['SentimentText'].apply(lambda x: function(x))
    _sk_clf = FirstColFtClassifier(wordNgrams=word_ngrams, thread=1, epoch=_iterations, lr=_lr, loss=_loss)
    _sk_clf.fit(_data_train[['SentimentText']], df_train['Sentiment'])
    _score = _sk_clf.score(_data_test[['SentimentText']], df_test['Sentiment'])
    print(f"Words ngrams: {word_ngrams}")
    return _score


index = 0
data = []
for loss in ["softmax", "ns", "hs"]:
    for epoch in [5, 10, 50, 100, 200]:
        for lr in [0.05, 0.1, 0.2, 0.5, 1]:
            start = time.time()
            scores = preprocess_train(data_train, data_test, [], _iterations=epoch, _lr=lr, _loss=loss, word_ngrams=3)
            end = time.time()
            print(f"Loss: {loss}, epoch: {epoch}, lr: {lr}")
            _time = end - start
            print(f"Time: {_time}")
            show_score(scores)
            data.append([index, loss, lr, epoch, np.mean(scores), time])
            index += 1


results_df = pd.DataFrame(data, columns=resultsColumns)
results_df.to_csv("Ex3ReportAmazon.csv", index=False, header=True)