from flask import Blueprint, request, jsonify
import pandas as pd
from app.service.dataExtraction import dataExtraction
from app.service.patterns import normalizeText, patterns

bp = Blueprint('classification', __name__)

@bp.route('/', methods=['POST'])
def getDataset():
    try:
        files = request.files.getlist('file')
    
        if not files or len(files) == 0:
            return jsonify({'erro': 'Nenhum arquivo enviado.'}), 400
        
        corpus = []

        for file in files:            
            df = dataExtraction(file, file.filename)
            records = df.to_dict(orient='records')
            corpus.extend(records)


        return jsonify({'mensagem': 'Arquivo recebido com sucesso.'}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
