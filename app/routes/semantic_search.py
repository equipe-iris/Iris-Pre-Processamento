from flask import Blueprint, request, jsonify, current_app
from app.service.semantic_search import semantic_search_service, sincronizar_novos_tickets

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
        
        results = semantic_search_service(query, model)
        if not results:
            return jsonify({'message': 'Nenhum resultado encontrado.'}), 404
        
        return jsonify({'results': results}), 200
        
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
        return jsonify({'error': str(e)}), 500

