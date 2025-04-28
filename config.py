import os
from dotenv import load_dotenv

load_dotenv()

PORT = os.getenv('PORT')
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS')
CLASSIFY_RESULTS_URL = os.getenv('CLASSIFY_RESULTS_URL')

