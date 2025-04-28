import joblib
from pathlib import Path


def predict_ticket(tokens, ticket):
    art = Path(__file__).resolve().parents[1] / 'artifacts'
    vectorizer_sent = joblib.load(art / 'sentiment_vectorizer.joblib')
    vectorizer_type = joblib.load(art / 'type_vectorizer.joblib')
    svm_sent   = joblib.load(art / 'sentiment_model.joblib')
    svm_type   = joblib.load(art / 'type_model.joblib')

    text = " ".join(tokens)

    vectorizer_predict = vectorizer_sent.transform([text])
    label_sentiment = svm_sent.predict(vectorizer_predict)[0]
    result_sentiment = label_sentiment.item()

    vectorizer_predict = vectorizer_type.transform([text])
    label_type = (svm_type.predict(vectorizer_predict)[0])
    result_type = label_type.item()

    return {
        'id': ticket['Id'],
        'title': ticket['Titulo'],
        'sentiment': result_sentiment,
        'type': result_type,
        'start_date': ticket['DataInicio'],
        'end_date': ticket['DataFinal']
    }