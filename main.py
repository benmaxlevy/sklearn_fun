# training dataset provided by: http://help.sentiment140.com/for-students
# text slang dataset converted by: https://www.convertcsv.com/html-table-to-csv.htm
# text slang dataset provided by: https://www.convertcsv.com/html-table-to-csv.htm
import os

import pandas as pd
import concurrent.futures
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from lib.spellcheck import spellcheck


def main(dataset_path, abbreviation_path):
    # importing the dataset - only keeping sentiment and tweet columns
    df = pd.read_csv(dataset_path, encoding='latin-1',
                     names=["sentiment", "w", "x", "y", "z", "tweet"])[["sentiment", "tweet"]]

    X = df["tweet"].tolist()
    y = df["sentiment"].tolist()

    # import the abbreviation dataset
    abbreviation = pd.read_csv(abbreviation_path, encoding='latin-1',
                               names=["abbreviation", "meaning"])

    abbreviations = abbreviation["abbreviation"].tolist()
    meanings = abbreviation["meaning"].tolist()

    # plz move this to a file
    def abbreviation_checker():
        """
        loop through X. for each word in X, check if it is in the abbreviation dictionary. if so, replace it with
        the expanded word
        """
        # loop through X
        for i in range(len(X)):
            print(i)
            # loop through each word in X
            for j in range(len(X[i])):
                if X[i][j] in abbreviations:
                    # replace word with expanded word
                    X[i] = X[i].replace(X[i][j], meanings[abbreviations.index(X[i][j])])

    #  multiprocessing spelling correction - takes way too long without - comment out to avoid absurd runtime
    # with concurrent.futures.ProcessPoolExecutor(os.cpu_count() - 12) as e:
    #     for i, result in enumerate(e.map(spellcheck, X, chunksize=10)):
    #         X[i] = result
    #         print(i)

    # multiprocessing abbreviation expansion - takes way too long without - comment out to avoid absurd runtime
    with concurrent.futures.ProcessPoolExecutor(os.cpu_count() - 12) as e:
        abbreviation_checker()
        print("cool")

    vect = CountVectorizer(stop_words="english")
    X = vect.fit_transform(X)

    # 70% training and 30% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=109)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    test = clf.predict(X_test)

    # return results from test
    return classification_report(y_test, test)


if __name__ == "__main__":
    print(main("dataset/training.1600000.processed.noemoticon.csv", "dataset/textslang.csv"))
