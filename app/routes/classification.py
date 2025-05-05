from flask import Blueprint, request, jsonify
import pandas as pd
import requests

from config import IA_CLASSIFY_RESULTS_URL
from app.service.data_extraction import data_extraction
from app.service.preprocessing import preprocessing
from app.service.training import sentiment_model, type_model
from app.service.predict import predict_ticket

bp = Blueprint('classification', __name__)

@bp.route('/predict', methods=['POST'])
def predict():
    try:
        files = request.files.getlist('file')
        file_ids = request.form.getlist('fileId')
    
        if not files or len(files) == 0:
            return jsonify({'erro': 'Nenhum arquivo enviado.'}), 400
        
        results = []
        for file, file_id in zip(files, file_ids):
            df = data_extraction(file, file.filename)
            records = df.to_dict(orient='records')

            processed = []
            for ticket in records:
                tokens = preprocessing(ticket['Interacao'])
                result = predict_ticket(tokens, ticket)
                processed.append(result)

            results.append({
                'file_id': file_id,
                'processed_tickets': processed
            })

        response = requests.post(IA_CLASSIFY_RESULTS_URL, json=results)
        if response.status_code != 200:
            
            return jsonify({
                'erro': 'Falha ao enviar resultados para /tickets/classification-results',
                'status_code': response.status_code,
                'response': response.text
            }), response.status_code         

        return jsonify('Dados processados com sucesso'), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@bp.route('/training', methods=['POST'])
def train():
    try:
        csv_path = ['app/dataset_training/dataset_treinamento_sentimento_v1.csv', 'app/dataset_training/dataset_treinamento_tipo_v1.csv']
        sentiment_df = pd.read_csv(csv_path[0], encoding='utf-8')
        type_df = pd.read_csv(csv_path[1], encoding='utf-8')
        
        documents = []
        labels = []
        
        for _, row in sentiment_df.iterrows():
            tokens = preprocessing(row['Interacao'])
            documents.append(" ".join(tokens))
            labels.append(row['Classe'])

        sentiment_result = sentiment_model(documents, labels)

        documents = []
        labels = []

        for _, row in type_df.iterrows():
            tokens = preprocessing(row['Interacao'])
            documents.append(" ".join(tokens))
            labels.append(row['Classe'])

        type_result = type_model(documents, labels)
        
        return jsonify({
            'Resultado treinamento de classificação dos sentimentos': sentiment_result,
            'Resultado treinamento de classificação de tipo': type_result
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
