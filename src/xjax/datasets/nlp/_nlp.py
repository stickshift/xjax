import pandas as pd
from nltk import sent_tokenize


_all__ = ["moby_dick", "simpsons"]


def moby_dick():
    text = open("melville-moby_dick.txt")
    sentences = sent_tokenize(text)
    return pd.DataFrame(sentences, column="sentences")


def simpsons():
    return pd.read_csv("simpsons_dataset.csv")

