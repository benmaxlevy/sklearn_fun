import pandas as pd

# import the abbreviation dataset
abbreviations = pd.read_csv("dataset/textslang.csv", encoding='latin-1',
                            names=["abbreviation", "meaning"])
abbreviations = dict(zip(abbreviations["abbreviation"], abbreviations["meaning"]))


def abbreviation_checker(X):
    """
    loop through X. for each word in X, check if it is in the abbreviation dictionary. if so, replace it with
    the expanded word
    """

    words = X.split()
    for j in range(len(words)):
        if words[j].upper() in abbreviations:
            words[j] = abbreviations[words[j].upper()]
    X = ' '.join(words)
    return X
