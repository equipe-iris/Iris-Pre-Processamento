import pandas as pd
from .utils import combineTextColumns

def dataExtraction():
    try:
        df = pd.read_csv('./dataset/Jira Exportar CSV (todos os campos) 20250312084318.csv', sep=';')
    except Exception as e:
        df = pd.read_csv('./dataset/Jira Exportar CSV (todos os campos) 20250312084318.csv')

    return combineTextColumns(df)