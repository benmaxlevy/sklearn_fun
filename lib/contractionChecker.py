from dataset.contractions import CONTRACTION_MAP


def contraction_checker(X):
    """
    loop through X. for each word in X, check if it corresponds with the key in the contractions dict. if so, replace it with the value in the contractions dict
    """
    words = X.split()
    for j in range(len(words)):
        if words[j].lower() in CONTRACTION_MAP:
            words[j] = CONTRACTION_MAP[words[j].lower()]
    X = ' '.join(words)
    return X
