import os
from dotenv import load_dotenv

load_dotenv()

IA_PORT = os.getenv('IA_PORT')
IA_ALLOWED_ORIGINS = os.getenv('IA_ALLOWED_ORIGINS')
IA_CLASSIFY_RESULTS_URL = os.getenv('IA_CLASSIFY_RESULTS_URL')
