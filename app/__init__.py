from flask import Flask
from flask_cors import CORS
from config import IA_ALLOWED_ORIGINS
from app.routes.classification import bp as classification_bp
from app.routes.sumarization import bp as sumarization_bp
from app.service.queue_consumer import run_consumer_thread

def createApp():
    app = Flask(__name__)
    
    # Habilita CORS para todas as rotas e origens
    CORS(app, origins=IA_ALLOWED_ORIGINS)

    # Importa e registra o blueprint de routes    
    app.register_blueprint(classification_bp, url_prefix='/classification')
    app.register_blueprint(sumarization_bp, url_prefix='/sumarization')
    
    # Inicia o consumidor de mensagens
    run_consumer_thread()

    return app
