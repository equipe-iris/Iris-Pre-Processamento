import pandas as pd

def dataExtraction(file, fileName):
    if fileName == 'Jira Exportar CSV (todos os campos) 20250312084318.csv':
        df = pd.read_csv(file, encoding='utf-8')
        selectColumn = ['Chave da item', 'Interacao']

        descriptionColumn = 'Descrição'
        commentColumn = 'Comentar'
        
        df['Interacao'] = df.apply(lambda row: joinInteraction(row, descriptionColumn, commentColumn), axis=1)
        finalDf = df[selectColumn]

        return finalDf
    df = pd.read_csv(file, delimiter=';', encoding='utf-8')
    selectColumn = ['ID', 'Interacao']

    descriptionColumn = 'Descrição'
    solutionColumn = 'Solução - Solução'

    df['Interacao'] = df.apply(lambda row: joinInteraction(row, descriptionColumn, solutionColumn), axis=1)
    finalDf = df[selectColumn]


    return finalDf

def joinInteraction(row, descriptionColumn, commentColumn):
    # Lista de colunas a ser juntada: "Descrição" + todas as colunas que começam com "Comentar"
    columns_to_join = [descriptionColumn] + [col for col in row.index if col.startswith(commentColumn)]

    # Para cada coluna, se o valor não for vazio ou nulo (após aplicar strip) ele é incluído
    values = [row[col].strip() for col in columns_to_join if pd.notnull(row[col]) and row[col].strip()]

    # Junta os valores com " | " apenas entre os textos não vazios
    return " | ".join(values)