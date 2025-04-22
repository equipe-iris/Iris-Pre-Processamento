import pandas as pd

def dataExtraction(file, fileName):
    if fileName == 'Jira Exportar CSV (todos os campos) 20250312084318.csv':
        df = pd.read_csv(file, encoding='utf-8')

        descriptionColumn = 'Descrição'
        commentColumn = 'Comentar'
        
        df['Interacao'] = df.apply(lambda row: joinInteraction(row, descriptionColumn, commentColumn), axis=1)

        finalDf = (df[['Resumo','Chave da item', 'Interacao']].rename(columns={'Resumo': 'Titulo', 'Chave da item': 'Id'}).copy())

        return finalDf
    df = pd.read_csv(file, delimiter=';', encoding='utf-8')

    descriptionColumn = 'Descrição'
    solutionColumn = 'Solução - Solução'

    df['Interacao'] = df.apply(lambda row: joinInteraction(row, descriptionColumn, solutionColumn), axis=1)

    finalDf = (df[['Título','ID', 'Interacao']].rename(columns={'Título': 'Titulo', 'ID': 'Id'}).copy())

    return finalDf

def joinInteraction(row, descriptionColumn, commentColumn):
    # Lista de colunas a ser juntada: "Descrição" + todas as colunas que começam com "Comentar"
    columns_to_join = [descriptionColumn] + [col for col in row.index if col.startswith(commentColumn)]

    # Para cada coluna, se o valor não for vazio ou nulo (após aplicar strip) ele é incluído
    values = []
    for col in columns_to_join:
        val = row[col]
        if pd.notnull(val):
            val = val.strip()
            if val:
                values.append(val)
    return " | ".join(values)
