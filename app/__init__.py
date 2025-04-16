from flask import Flask
from flask_cors import CORS
from config import ALLOWED_ORIGINS

def createApp():
    app = Flask(__name__)
    
    # Habilita CORS para todas as rotas e origens
    CORS(app, origins=ALLOWED_ORIGINS)

    # Importa e registra o blueprint de routes
    from app.routes.preprocessingRoute import bp as preprocessing_bp
    app.register_blueprint(preprocessing_bp, url_prefix='/preprocessing')

    return app