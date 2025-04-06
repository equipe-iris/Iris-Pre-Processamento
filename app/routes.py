from flask import Blueprint, request, jsonify
import pandas as pd
from app.dataExtraction import dataExtraction

bp = Blueprint('routes', __name__)

@bp.route('/get-dataset', methods=['POST'])
async def get_dataset():
    try:
        text = await dataExtraction.dataExtraction()
        return jsonify({
            "message": "Arquivo processado com sucesso."
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
