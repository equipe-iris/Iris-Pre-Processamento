from app.service.patterns import normalize_text, patterns
from app.service.tokenization import tokenization
from app.service.stopword import remove_stopwords
from app.service.stemming import stemming


def preprocessing(text):
    normalized_text = normalize_text(text, patterns)
    tokens = tokenization(normalized_text)
    filtered_tokens = remove_stopwords(tokens)
    stemming_tokens = stemming(filtered_tokens)
    
    return stemming_tokens