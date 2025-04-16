from flask import Blueprint, request, jsonify
from app.service.dataExtraction import dataExtraction
from app.service.preprocessing import preprocessing

bp = Blueprint('preprocessing', __name__)

@bp.route('/upload-dataset', methods=['POST'])
def getDataset():
    try:
        if 'file' not in request.files:
            return jsonify({'erro': 'Nenhum arquivo enviado.'}), 400

        file = request.files['file']

        print(file.filename)

        if file.filename != 'Jira Exportar CSV (todos os campos) 20250312084318.csv' and file.filename != 'Chamados Porto.csv':
            return jsonify({'erro': 'Nome do arquivo inválido.'}), 400
        
        df = dataExtraction(file, file.filename)

        print(df)
        preProcessDf = preprocessing(df)

        return jsonify({'mensagem': 'Arquivo recebido com sucesso.'}), 200
        

    except Exception as e:
        return jsonify({'error': str(e)}), 500
