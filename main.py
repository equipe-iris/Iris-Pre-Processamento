import os
from dotenv import load_dotenv
from app import app

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

if __name__ == '__main__':
    # Recupera a variável "PORT" do ambiente e define um valor padrão caso ela não esteja configurada
    port = int(os.getenv("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)