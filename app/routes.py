from flask import Blueprint, request, jsonify
import pandas as pd

bp = Blueprint('routes', __name__)

@bp.route('/get-dataset', methods=['POST'])
def get_dataset():
    try:
        if 'file' not in request.files:
            return jsonify({'erro': 'Nenhum arquivo enviado.'}), 400

        file = request.files['file']

        if file.filename != 'Jira Exportar CSV (todos os campos) 20250312084318.csv' or file.name != 'Chamados Porto.csv':
            return jsonify({'erro': 'Nome do arquivo inválido.'}), 400
        return jsonify({'mensagem': 'Arquivo recebido com sucesso.'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
