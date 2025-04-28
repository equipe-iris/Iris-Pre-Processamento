from nltk.corpus import stopwords

stopwords_pt = set(stopwords.words('portuguese'))

relevant_stopwords = {'não', 'mas', 'ainda', 'já', 'só', 'que'}

custom_stopwords = stopwords_pt - relevant_stopwords

def remove_stopwords(tokens):
    return [token for token in tokens if token not in custom_stopwords]