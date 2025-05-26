from app import createApp
from config import IA_PORT
import nltk

app = createApp()

def dependencies_dowload():
    nltk.download('punkt')
    nltk.download('stopwords')


if __name__ == '__main__':
    dependencies_dowload()
    port = IA_PORT
    app.run(debug=False, host='0.0.0.0', port=port)
