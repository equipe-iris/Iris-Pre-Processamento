from flask import Flask
from flask_cors import CORS
from config import ALLOWED_ORIGINS

def createApp():
    app = Flask(__name__)
    
    # Habilita CORS para todas as rotas e origens
    CORS(app, origins=ALLOWED_ORIGINS)

    # Importa e registra o blueprint de routes
    from app.routes.classification import bp as classification_bp
    app.register_blueprint(classification_bp, url_prefix='/classification')

    return app