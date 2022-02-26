from spellchecker import SpellChecker

if __name__ == "main":
    raise Exception("This file is not intended to be run directly")


def spellcheck(text):
    # lowercase the text (even though the vectorizer will do it later)
    text = text.lower()
    spell = SpellChecker()
    # get unknown words
    misspelled = spell.unknown(text.split())
    # loop through text (by space)
    for word in (text.split()):
        if word in misspelled:
            # if word is misspelled, replace it with correct word
            text = text.replace(word, spell.correction(word))
            print(text)
    return text
