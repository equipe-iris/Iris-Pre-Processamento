from flask import Blueprint, request, jsonify
import pandas as pd

from app.service.summarization_training import summarization_training
from app.service.summarization_predict import predict_tickets

bp = Blueprint('sumarization', __name__)


@bp.route('/training', methods=['POST'])
def train():
    try:
        df = pd.read_csv('app/dataset_training/training_sumarization.csv', encoding='utf-8')
        result = summarization_training(df)
        print(result)
        return jsonify({'Resultado do treinamento': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@bp.route('predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        summary = predict_tickets(text)
        print(f'Resultado do modelo ${summary}')
        return jsonify({'summary': summary}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500