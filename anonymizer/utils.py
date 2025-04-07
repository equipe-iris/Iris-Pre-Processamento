import string
from typing import List
from presidio_analyzer import RecognizerResult

# Consts para facilitar a filtragrem dos textos
ACRONYMS = {
    "CPF", "CPF:" "CLT", "CNPJ", "OS"
}

PROBLEMATIC_TERMS = {
    "Bom Dia", "Olá", "Solicito", "Peço", "Gentileza", "Assunto", "RE", "Bom", "Boa", "At.te", "Telefone", "Fevereiro"
}

ENTITY_TYPES_TO_EXCLUDE = {"DATE_TIME"}

# --------------------------------------------------------------

# Mostrar os resultados do analisador de forma legível
def print_analyzer_results(results: List[RecognizerResult], text: str):
    """Print the results in a human readable way."""

    for i, result in enumerate(results):
        print(f"Result {i}:")
        print(f" {result}, text: {text[result.start:result.end]}")
        print(f" {result.entity_type}, score: {result.score}")

        if result.analysis_explanation is not None:
            print(f" {result.analysis_explanation.textual_explanation}")

# Descapitalizar palavras todas maiúsculas
def normalize_all_uppercase_words(text):
    lines = text.splitlines() 
    normalized_lines = []

    # Separa o texto em linhas para manter as quebras de linha
    for line in lines:
        words = line.split()
        
        normalized_words = []
        for word in words:
            # Remove possíveis caracteres especiais e pontuação
            cleaned_word = word.strip(string.punctuation)

            # Verifica se a palavra está em maiúsculas e não é um acrônimo listado
            if cleaned_word.isupper() and cleaned_word not in ACRONYMS:
                normalized_words.append(cleaned_word.capitalize())
            else:
                normalized_words.append(word)

        normalized_lines.append(' '.join(normalized_words))

    return '\n'.join(normalized_lines)  # Recompõe o texto com as quebras de linha originais

# Filtrar termos problemáticos
def filter_problematic_terms(results: List[RecognizerResult], text: str):
    filtered_results = []
    
    normalized_terms = {t.lower().strip() for t in PROBLEMATIC_TERMS}  # Normalizar os termos já listados
    for result in results:
        term = text[result.start:result.end].strip().lower()  # Normalizar o termo encontrado

        # Verificar se o termo está nos termos problemáticos
        if term not in normalized_terms:
            filtered_results.append(result)

    return filtered_results

# Exclui tipos de entidades indesejadas
def exclude_entity_types(results: List[RecognizerResult]):
    return [result for result in results if result.entity_type not in ENTITY_TYPES_TO_EXCLUDE]