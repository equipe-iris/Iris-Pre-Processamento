from presidio_anonymizer import OperatorConfig

# Operadores personalizados para anonimização
custom_operators = {
    "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<ENDERECO_EMAIL>"}),
    "PERSON": OperatorConfig("replace", {"new_value": "<NOME>"}),
    "LOCATION": OperatorConfig("replace", {"new_value": "<LOCAL>"}),
}