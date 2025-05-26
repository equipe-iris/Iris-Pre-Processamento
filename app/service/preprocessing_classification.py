from app.service.patterns import normalize_text, patterns
from app.service.tokenization import tokenization
from app.service.stopword import remove_stopwords
from app.service.stemming import stemming


def preprocessing_classification(text):
    tokens = tokenization(text)
    filtered_tokens = remove_stopwords(tokens)
    stemming_tokens = stemming(filtered_tokens)
    
    return stemming_tokens