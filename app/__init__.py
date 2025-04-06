from flask import Flask

app = Flask(__name__)

# Importa e registra o blueprint de routes
from app.routes import bp as routes_bp
app.register_blueprint(routes_bp)