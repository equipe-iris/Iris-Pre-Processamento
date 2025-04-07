from presidio_anonymizer import AnonymizerEngine
from custom_operators import custom_operators

anonymizer = AnonymizerEngine()

# Anonimiza o texto com base nos resultados da análise
def anonymize_text(text, analyzer_results):
    anonymized_text = anonymizer.anonymize(
        text=text,
        analyzer_results=analyzer_results,
        operators=custom_operators, # Operadores personalizados. Atualmente, fazem a tradução de algumas entidades para o português
    )
    return anonymized_text.text