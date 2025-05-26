from flask import Blueprint, request, jsonify, current_app
from app.service.semantic_search import semantic_search_service, sincronizar_novos_tickets
from config import IA_SEMANTIC_SEARCH_URL
import requests

bp = Blueprint('semantic-search', __name__)

@bp.route('/search', methods=['POST'])
def semantic_search():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Consulta não fornecida.'}), 400

        query = data['query']
        model = current_app.semantic_search_model
        if not model:
            return jsonify({'error': 'Modelo de busca semântica não carregado.'}), 500
        
        results = semantic_search_service(query, model, 10)
        if not results:
            return jsonify({'message': 'Nenhum resultado encontrado.'}), 404
        
        print(results)
        
        try:
            response = requests.post(IA_SEMANTIC_SEARCH_URL, json=results)
            response.raise_for_status()
            return jsonify(response.json()), response.status_code
        
        except requests.RequestException as e:
            print(f"Erro ao enviar resultados para {IA_SEMANTIC_SEARCH_URL}: {e}")
            return jsonify({'error': 'Erro ao enviar resultados para o endpoint de busca semântica.'}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/sync', methods=['POST'])
def sync_semantic_index():
    try:
        print("Sincronizando índice semântico com novos tickets...")
        
        data = request.get_json()
        novos_tickets = data.get("tickets", [])
        model = current_app.semantic_search_model
        if not model:
            return jsonify({'error': 'Modelo não carregado.'}), 500

        sincronizar_novos_tickets(novos_tickets, model)
        return jsonify({'message': 'Índice atualizado com sucesso.'}), 200
    except Exception as e:
        print("erro", e)
        return jsonify({'error': str(e)}), 500

