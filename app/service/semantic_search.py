import requests
import pandas as pd
import faiss
from app.service.semantic_state import INDEX_FAISS, TEXTOS, EMBEDDINGS, INDEX_LOCK
import numpy as np

CSV_CAMINHO = "app/service/busca semantica/dataset_so_interacao.csv"
COLUNA_DESCRICAO = "Descrição" 

def carregar_descricoes(csv_path=CSV_CAMINHO, coluna=COLUNA_DESCRICAO):
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8", engine="python", on_bad_lines="warn", quoting=3)
    if coluna not in df.columns:
        raise ValueError(f"Coluna '{coluna}' não encontrada no CSV.")
    df[coluna] = df[coluna].fillna("").str.strip()
    df = df[df[coluna] != ""]
    df = df[[coluna]].drop_duplicates().reset_index(drop=True)
    return df[coluna].tolist()

def construir_indice_faiss(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def buscar_descricoes():
    try:
        response = requests.get(
            "http://localhost:7000/tickets/processed-tickets?start_date=&end_date=", 
            timeout=10
        )
        response.raise_for_status()
        tickets = response.json()
        return [
            {
                "id": t["id"],
                "title": t["title"],
                "content": t["content"]
            }
            for t in tickets
        ]
    except Exception as e:
        print(f"Erro ao buscar textos do endpoint: {e}")
        return []

def inicializar_indice_e_embeddings(model):
    print("Inicializando índice FAISS e embeddings...")
    global INDEX_FAISS, TEXTOS, EMBEDDINGS

    tickets = buscar_descricoes()
    if not tickets:
        print("Nenhum ticket encontrado no endpoint, inicializando vazio.")
        tickets = []

    textos_para_embedding = [f"{t['title']} {t['content']}" for t in tickets]

    embeddings = model.encode(textos_para_embedding, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True) if textos_para_embedding else None
    index = construir_indice_faiss(embeddings) if embeddings is not None else None

    with INDEX_LOCK:
        TEXTOS.clear()
        TEXTOS.extend(tickets)
        EMBEDDINGS = embeddings
        INDEX_FAISS = index

def semantic_search_service(query, model, top_k=5):

    with INDEX_LOCK:
        if INDEX_FAISS is None or not TEXTOS:
            return []

        embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        distances, indices = INDEX_FAISS.search(embedding, top_k)

        resultados = []
        usados = set()
        for i, idx in enumerate(indices[0]):
            if idx < len(TEXTOS):
                ticket = TEXTOS[idx]
                ticket_id = ticket.get("id")
                if ticket_id not in usados:
                    usados.add(ticket_id)
                    resultados.append({
                        "id": ticket.get("id"),
                        "score": float(distances[0][i])
                    })
                if len(resultados) >= top_k:
                    break
        return resultados

def sincronizar_novos_tickets(novos_tickets, model):
    global EMBEDDINGS, INDEX_FAISS, TEXTOS

    if not novos_tickets:
        return print("Nenhum novo ticket para sincronizar.")

    textos_para_embedding = [f"{t['title']} {t['content']}" for t in novos_tickets]
    novos_embeddings = model.encode(textos_para_embedding, convert_to_numpy=True, normalize_embeddings=True)

    with INDEX_LOCK:
        
        TEXTOS.extend(novos_tickets)
        if EMBEDDINGS is not None:
            EMBEDDINGS = np.vstack([EMBEDDINGS, novos_embeddings])
        else:
            EMBEDDINGS = novos_embeddings

        if INDEX_FAISS is not None:
            INDEX_FAISS.add(novos_embeddings)
        else:
            INDEX_FAISS = construir_indice_faiss(EMBEDDINGS)

