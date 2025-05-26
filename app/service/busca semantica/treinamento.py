import pandas as pd
import random
import os
import logging
import argparse
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, models
from torch.utils.data import DataLoader
import faiss
import numpy as np
import shutil

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Parâmetros globais
MODELO_SALVO = "modelo_semantico_treinado"
INDEX_SALVO = "indice_faiss.index"
CSV_CAMINHO = "dataset_so_interacao.csv"
CHECKPOINT_DIR = "checkpoints2"
METRICAS_CSV = "metricas_treinamento.csv"
SEED = 42
BEST_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, "best")

def set_seed(seed):
    """Garante reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def gerar_pares_artificiais(df, coluna_texto="Descrição"):
    """
    Gera pares positivos (texto, texto) e negativos (texto, outro_texto).
    """
    textos = df[coluna_texto].tolist()
    exemplos = []
    for i, original in enumerate(textos):
        exemplos.append(InputExample(texts=[original, original], label=1.0))
        # Par negativo: sorteia outro texto diferente
        negativo_idx = random.choice([j for j in range(len(textos)) if j != i])
        negativo = textos[negativo_idx]
        exemplos.append(InputExample(texts=[original, negativo], label=0.0))
    return exemplos

def carregar_dados_csv(csv_path=CSV_CAMINHO):
    """
    Carrega e limpa o CSV, retorna DataFrame apenas com a coluna 'Descrição'.
    """
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8", engine="python", on_bad_lines="warn", quoting=3)
    df = df.sample(frac=0.5, random_state=SEED) 
    if "Descrição" not in df.columns:
        raise ValueError("Coluna 'Descrição' não encontrada no CSV.")
    df["Descrição"] = df["Descrição"].fillna("").str.strip()
    df = df[df["Descrição"] != ""]
    df = df[["Descrição"]].drop_duplicates().reset_index(drop=True)
    print(df.head(10))  # Debug: veja as primeiras linhas
    return df

def split_train_val(df, val_frac=0.1):
    """
    Divide o DataFrame em treino e validação.
    """
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    val_size = int(len(df) * val_frac)
    return df.iloc[val_size:], df.iloc[:val_size]

def treinar_modelo(df_train, df_val, epochs, patience=3):
    """
    Treina o modelo SentenceTransformer com early stopping e salva métricas.
    """
    # Inicializa modelo
    if os.path.exists(CHECKPOINT_DIR) and os.path.exists(os.path.join(CHECKPOINT_DIR, "config.json")):
        logging.info("🔁 Carregando modelo salvo anteriormente...")
        model = SentenceTransformer(CHECKPOINT_DIR)
    else:
        logging.info("🧠 Criando novo modelo do zero...")
        word_embedding_model = models.Transformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    exemplos_train = gerar_pares_artificiais(df_train)
    exemplos_val = gerar_pares_artificiais(df_val)
    dataloader = DataLoader(exemplos_train, shuffle=True, batch_size=4)
    loss = losses.CosineSimilarityLoss(model)
    evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(exemplos_val, name='avaliacao')

    # Early stopping manual
    best_score = -np.inf
    best_epoch = 0
    patience_counter = 0
    metricas = []

    for epoch in range(1, epochs + 1):
        logging.info(f"🚀 Iniciando epoch {epoch}/{epochs}")
        model.fit(
            train_objectives=[(dataloader, loss)],
            epochs=1,
            warmup_steps=5,
            show_progress_bar=True
        )
        score = evaluator(model)
        # Exemplo de extração da métrica cosine_similarity
        cosine_score = score['cosine_similarity'] if 'cosine_similarity' in score else list(score.values())[0]
        logging.info(f"📊 Epoch {epoch}: Validação (CosineSimilarity) = {cosine_score:.4f}")
        metricas.append({"epoch": epoch, "val_score": cosine_score})

        # Early stopping
        if cosine_score > best_score:
            best_score = cosine_score
            best_epoch = epoch
            patience_counter = 0
            # Limpa o diretório antes de salvar para evitar erro safetensors
            if os.path.exists(BEST_CHECKPOINT_DIR):
                shutil.rmtree(BEST_CHECKPOINT_DIR)
            model.save(BEST_CHECKPOINT_DIR)
            logging.info(f"💾 Novo melhor modelo salvo (epoch {epoch})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"⏹️ Early stopping ativado em epoch {epoch}")
                break

    # Salva métricas em CSV
    pd.DataFrame(metricas).to_csv(METRICAS_CSV, index=False)
    logging.info(f"📈 Métricas salvas em {METRICAS_CSV}")

    # Carrega o melhor modelo salvo
    model = SentenceTransformer(BEST_CHECKPOINT_DIR)
    return model

def construir_indice_faiss(model, textos):
    """
    Gera embeddings e constrói índice FAISS.
    """
    embeddings = model.encode(textos, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_SALVO)
    logging.info(f"✅ Índice FAISS salvo como {INDEX_SALVO}")
    return index

def modo_pesquisa_interativo(model, index, textos):
    """
    Interface de busca semântica interativa.
    Só busca na coluna 'Descrição' e evita duplicatas.
    """
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
    set_seed(SEED)
    parser = argparse.ArgumentParser(description="Treinamento e busca semântica.")
    parser.add_argument("--epochs", type=int, default=10, help="Número máximo de epochs para treinamento")
    parser.add_argument("--val_frac", type=float, default=0.1, help="Fração de validação")
    parser.add_argument("--patience", type=int, default=3, help="Paciencia para early stopping")
    args = parser.parse_args()

    df = carregar_dados_csv()
    logging.info(f"📄 {len(df)} textos carregados.")

    df_train, df_val = split_train_val(df, val_frac=args.val_frac)
    logging.info(f"🔗 {len(df_train)} exemplos de treino, {len(df_val)} de validação.")

    model = treinar_modelo(df_train, df_val, args.epochs, patience=args.patience)
    index = construir_indice_faiss(model, df["Descrição"].tolist())
    modo_pesquisa_interativo(model, index, df["Descrição"].tolist())

if __name__ == "__main__":
    main()
