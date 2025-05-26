from flask import Blueprint, jsonify
import pandas as pd

from app.service.classification_training import sentiment_model, type_model
from app.service.preprocessing_classification import normalize_text

bp = Blueprint('classification', __name__)

@bp.route('/training', methods=['POST'])
def train():
    try:
        csv_path = ['app/dataset_training/dataset_treinamento_sentimento_v1.csv', 'app/dataset_training/dataset_treinamento_tipo_v1.csv']
        sentiment_df = pd.read_csv(csv_path[0], encoding='utf-8')
        type_df = pd.read_csv(csv_path[1], encoding='utf-8')
        
        documents = []
        labels = []
        
        for _, row in sentiment_df.iterrows():
            tokens = sentiment_model(row['Interacao'])
            documents.append(" ".join(tokens))
            labels.append(row['Classe'])

        sentiment_result = sentiment_model(documents, labels)

        documents = []
        labels = []

        for _, row in type_df.iterrows():
            tokens = type_model(row['Interacao'])
            documents.append(" ".join(tokens))
            labels.append(row['Classe'])

        type_result = type_model(documents, labels)
        
        return jsonify({
            'Resultado treinamento de classificação dos sentimentos': sentiment_result,
            'Resultado treinamento de classificação de tipo': type_result
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
