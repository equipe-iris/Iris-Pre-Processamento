import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Parâmetros globais
BEST_CHECKPOINT_DIR = os.path.join("checkpoints2", "best")
CSV_CAMINHO = "dataset_so_interacao.csv"  
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

def modo_pesquisa_interativo(model, index, textos):
    print("\n🧠 Modo de busca semântica. Digite 'sair' para encerrar.\n")
    while True:
        consulta = input("🔍 Digite sua pesquisa: ").strip()
        if consulta.lower() in ["sair", "exit", "quit"]:
            print("👋 Encerrando pesquisa.")
            break
        embedding = model.encode([consulta], convert_to_numpy=True, normalize_embeddings=True)
        distances, indices = index.search(embedding, 5)
        resultados = []
        usados = set()
        print("\n🔎 Resultados:")
        for i, idx in enumerate(indices[0]):
            if idx < len(textos):
                texto = textos[idx]
                if texto not in usados:
                    usados.add(texto)
                    print(f"{len(resultados)+1}. {texto} (similaridade: {distances[0][i]:.4f})")
                    resultados.append(texto)
                if len(resultados) >= 5:
                    break
        if not resultados:
            print("Nenhum resultado encontrado.")
        print("-" * 40)

def main():
    # Carrega o modelo treinado
    model = SentenceTransformer(BEST_CHECKPOINT_DIR)

    # Carrega descrições do novo dataset
    textos = carregar_descricoes()

    # Gera embeddings para o novo dataset
    embeddings = model.encode(textos, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)

    # Cria índice FAISS para busca rápida
    index = construir_indice_faiss(embeddings)

    # Inicia modo de busca
    modo_pesquisa_interativo(model, index, textos)

if __name__ == "__main__":
    main()