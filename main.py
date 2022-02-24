# dataset provided by: http://help.sentiment140.com/for-students

import pandas as pd
import multiprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from lib.spellcheck import spellcheck

if __name__ == "__main__":
    # importing the dataset - only keeping sentiment and tweet columns
    df = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding='latin-1',
                     names=["sentiment", "w", "x", "y", "z", "tweet"])[["sentiment", "tweet"]]

    X = df["tweet"].tolist()
    y = df["sentiment"].tolist()


    # preprocessing the dataset - spelling correction
    def pp(i):
        # call spellcheck function and replace value with corrected value
        X[i] = spellcheck(X[i])

    # multiprocessing spelling correction - takes way too long without
    pool = multiprocessing.Pool()
    pool.map(pp, range(len(X)))
    pool.join()
    pool.close()

    vect = CountVectorizer(stop_words="english")
    X = vect.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=109)  # 70% training and 30% test

    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    test = clf.predict(X_test)

    # print results from test
    print(classification_report(y_test, test))
