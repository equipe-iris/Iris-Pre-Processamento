from analyzer_config import analyze_text
from anonymizer_config import anonymize_text

from utils import normalize_all_uppercase_words, filter_problematic_terms, exclude_entity_types
from chamados import chamados

# Definição do texto para anonimização
sample_text = chamados[0] # Substituir o índice para testar outros textos
normalized_sample = normalize_all_uppercase_words(sample_text)

# Análise do texto para detectar entidades sensíveis
results = analyze_text(normalized_sample)

# Filtragem em cima dos resultados da análise
filtered_results = exclude_entity_types(results)
filtered_results = filter_problematic_terms(filtered_results, normalized_sample)

# Anonimização do texto com base nas entidades detectadas
anonymized_text = anonymize_text(normalized_sample, filtered_results)
print("\n---Texto Anonimizado ---")
print(anonymized_text)