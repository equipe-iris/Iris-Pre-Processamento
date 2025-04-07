from presidio_analyzer import Pattern, PatternRecognizer

# Regex patterns para CPF e Telefones no formato brasileiro
custom_patterns = [
    {
        "entity_type": "CPF",
        "patterns": [
            Pattern("cpf_com_pontuacao", r"(\d{3}\.\d{3}\.\d{3}-\d{2})", 0.95),
            Pattern("cpf_sem_pontuacao", r"(\d{11})", 0.5),
        ],
        "context": ["cpf", "documento", "registro", "cadastro"]
    },
    {
        "entity_type": "TELEFONE",
        "patterns": [
            Pattern("celular_com_espaco_com_especiais", r"\(\d{2}\)\s9\d{4}-\d{4}", 0.9),
            Pattern("celular_sem_espaco_com_especiais", r"\(\d{2}\)9\d{4}-\d{4}", 0.9),
            Pattern("celular_com_espaco_sem_especiais", r"\d{2}\s9\d{8}", 0.85),
            Pattern("celular_sem_espaco_sem_especiais", r"\d{2}9\d{8}", 0.5),
            Pattern("fixo_com_espaco_com_especiais", r"\(\d{2}\)\s\d{4}-\d{4}", 0.9),
            Pattern("fixo_sem_espaco_com_especiais", r"\(\d{2}\)\d{4}-\d{4}", 0.9),
            Pattern("fixo_com_espaco_sem_especiais", r"\d{2}\s\d{8}", 0.85),
            Pattern("fixo_sem_espaco_sem_especiais", r"\d{2}\d{8}", 0.7),
        ],
        "context": ["telefone", "Telefone" "celular", "contato", "contatos"]  
    }
]