from app import createApp
from config import PORT

app = createApp()


if __name__ == '__main__':
    # Recupera a variável "PORT" do ambiente e define um valor padrão caso ela não esteja configurada
    port = PORT
    app.run(debug=True, host='0.0.0.0', port=port)