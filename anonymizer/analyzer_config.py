from presidio_analyzer import AnalyzerEngine, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider

from custom_patterns import custom_patterns

# Configurações iniciais, definição de idioma, carregamento do modelo
configuration = {
    "nlp_engine_name": "spacy",
    "models": [{ "lang_code": "pt", "model_name": "pt_core_news_lg" }]
}

provider = NlpEngineProvider(nlp_configuration=configuration)
nlp_engine = provider.create_engine()

analyzer = AnalyzerEngine(
    nlp_engine=nlp_engine, 
    supported_languages=["pt"]
)

# Adição de padrões de reconhecimento personalizados
for pattern in custom_patterns:
    recognizer = PatternRecognizer(
        supported_language="pt",
        supported_entity=pattern["entity_type"],
        patterns=pattern["patterns"],
        context=pattern["context"],
    )
    analyzer.registry.add_recognizer(recognizer)


# Adição de palavras de contexto adicionais para o reconhecedor de pessoas
recognizers = analyzer.registry.get_recognizers(language="pt", entities=["PERSON"])
person_recognizer = next(
        r for r in recognizers
        if "PERSON" in r.supported_entities
    )

person_recognizer.context = list(set(person_recognizer.context + [
    "nome", "usuario", "pessoa", "colaborador", "colaboradora",
    "funcionario", "funcionaria", "atenciosamente", "cadastro",
    "exclusao", "cc", "clt", "de", "para"
]))

# Analiza o texto e retorna as entidades reconhecidas
def analyze_text(text):
    results = analyzer.analyze(
        text=text,
        language="pt",
        score_threshold=0.7,
    )

    return results