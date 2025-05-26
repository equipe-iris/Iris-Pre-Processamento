from nltk import word_tokenize

def tokenization(text):
    text = text.lower()
    tokens = word_tokenize(text, language = 'portuguese')
    
    importantMarks = {'?'}

    filtredTokens = [token for token in tokens if token.isalpha() or token in importantMarks]

    return filtredTokens
