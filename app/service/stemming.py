from nltk.stem import RSLPStemmer

stemmer = RSLPStemmer()


def stemming(tokens):
    return [stemmer.stem(token) for token in tokens]