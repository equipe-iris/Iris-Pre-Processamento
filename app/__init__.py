from flask import Flask
from flask_cors import CORS
from config import IA_ALLOWED_ORIGINS
from app.routes.classification import bp as classification_bp
from app.routes.semantic_search import bp as semantic_search_bp
from sentence_transformers import SentenceTransformer
from app.service.semantic_search import inicializar_indice_e_embeddings

SEMANTIC_SEARCH_MODEL_DIR = 'app/artifacts/semantic_search'

def createApp():
    app = Flask(__name__)
    
    # Habilita CORS para todas as rotas e origens
    CORS(app, origins=IA_ALLOWED_ORIGINS)

    # Carrega o modelo de busca semântica
    semantic_search_model = SentenceTransformer(SEMANTIC_SEARCH_MODEL_DIR)
    app.semantic_search_model = semantic_search_model

    # Inicializa o índice FAISS e os embeddings
    inicializar_indice_e_embeddings(semantic_search_model)

    # Registra o blueprint de routes
    app.register_blueprint(classification_bp, url_prefix='/classification')
    app.register_blueprint(semantic_search_bp, url_prefix='/semantic-search')

    return app