import pandas as pd

# Caminho para o seu arquivo original
caminho_csv = "datasetdois.csv"

# Carrega o dataset corretamente usando vírgula como separador
df = pd.read_csv(caminho_csv, sep=",", encoding="utf-8")

# Verifica se a coluna 'interacao' existe
if "interacao" not in df.columns:
    raise ValueError("A coluna 'interacao' não foi encontrada no dataset.")

# Mantém apenas a coluna 'interacao'
df = df[["interacao"]].copy()

# Salva em um novo arquivo CSV
df.to_csv("dataset_so_interacao.csv", index=False)

print("✅ Dataset salvo como dataset_so_interacao.csv (apenas a coluna 'interacao')")
