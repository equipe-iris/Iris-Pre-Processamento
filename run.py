from app import createApp
from config import IA_PORT

app = createApp()


if __name__ == '__main__':
    # Recupera a variável "IA_PORT" do ambiente e define um valor padrão caso ela não esteja configurada
    port = IA_PORT
    app.run(debug=True, host='0.0.0.0', port=port)
