import threading

INDEX_FAISS = None
TEXTOS = []
EMBEDDINGS = None
INDEX_LOCK = threading.Lock()