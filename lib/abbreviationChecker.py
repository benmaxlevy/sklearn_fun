def abbreviation_checker(X, abbreviations, meanings):
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