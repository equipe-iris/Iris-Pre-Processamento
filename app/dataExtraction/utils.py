import pandas as pd

# Função auxiliar para verificar se um valor não é vazio ou nulo
def isNotEmpty(text):
    if pd.isnull(text):
        return False
    if isinstance(text, str):
        return text.strip() != ""
    return True

import pandas as pd

def combineTextColumns(dataFrame):
    # Lista de colunas de descrição e obtenção das colunas que contenham "Comentar"
    descriptionColumns = ["ID da item", "Descrição"]
    commentColumns = [col for col in dataFrame.columns if "Comentar" in col]

    # Função para combinar os textos não vazios de uma linha
    def combineRow(row):
        texts = []
        # Itera sobre as colunas de descrição e adiciona o texto se houver conteúdo
        for column in descriptionColumns:
            if isNotEmpty(row.get(column, "")):
                texts.append(str(row[column]).strip())
        # Itera sobre as colunas "Comentar" e adiciona o texto se houver conteúdo
        for column in commentColumns:
            if isNotEmpty(row.get(column, "")):
                texts.append(str(row[column]).strip())
        # Junta os textos não vazios usando '|' como separador
        return "|".join(texts)
    
    # Cria a nova coluna "Iteração"
    dataFrame["Iteração"] = dataFrame.apply(combineRow, axis=1)

    # Retorna um array contendo apenas os textos da coluna "Iteração"
    return dataFrame["Iteração"].tolist()