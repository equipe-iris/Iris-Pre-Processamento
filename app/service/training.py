from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib


def sentiment_model(documents, labels):
    docs_train, docs_test, y_train, y_test = train_test_split(
        documents,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10_000)
    X_train = vectorizer.fit_transform(docs_train)
    X_test  = vectorizer.transform(docs_test)

    model = SVC(
        kernel='linear',
        probability=True,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    joblib.dump(vectorizer, 'app/artifacts/sentiment_vectorizer.joblib')
    joblib.dump(model, 'app/artifacts/sentiment_model.joblib')

    return {        
        'mensagem': 'Treinamento concluído e modelo SVM salvo.',
        'accuracy': accuracy,
        'classification_report': report_dict
    }

def type_model(documents, labels):
    docs_train, docs_test, y_train, y_test = train_test_split(
        documents,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10_000)
    X_train = vectorizer.fit_transform(docs_train)
    X_test  = vectorizer.transform(docs_test)

    model = SVC(
        kernel='linear',
        probability=True,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    joblib.dump(vectorizer, 'app/artifacts/type_vectorizer.joblib')
    joblib.dump(model, 'app/artifacts/type_model.joblib')

    return {        
        'mensagem': 'Treinamento concluído e modelo SVM salvo.',
        'accuracy': accuracy,
        'classification_report': report_dict
    }
