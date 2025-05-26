import pandas as pd
from datetime import datetime
from app.service.patterns import normalize_text, patterns


def data_extraction(file, file_name):
    if file_name == 'Jira Exportar CSV (todos os campos) 20250312084318.csv':
        df = pd.read_csv(file, encoding='utf-8')

        df['DataInicio'] = df['Criado'].apply(safe_format_jira)
        df['DataFinal']  = df['Resolvido'].apply(safe_format_jira)
        df['Responsavel'] = df['Responsável']

        df['Interacao'] = df.apply(
            lambda row: join_interaction(row, 'Descrição', 'Comentar'),
            axis=1
        )

        final_df = (
            df[['Resumo', 'Chave da item', 'Responsavel', 'Interacao', 'DataInicio', 'DataFinal']]
            .rename(columns={'Resumo': 'Titulo', 'Chave da item': 'Id'})
            .copy()
        )

        return final_df

    df = pd.read_csv(file, delimiter=',', encoding='utf-8')
    df['Data de abertura'].head()
    df['DataInicio'] = df['Data de abertura'].apply(safe_format_porto)
    df['DataFinal']  = df['Data de fechamento'].apply(safe_format_porto)
    df['Responsavel'] = df['Atribuído para - Técnico']

    df['Interacao'] = df.apply(
        lambda row: join_interaction(row, 'Descrição', 'Solução - Solução'),
        axis=1
    )

    final_df = (
        df[['Título', 'ID', 'Responsavel', 'Interacao', 'DataInicio', 'DataFinal']]
        .rename(columns={'Título': 'Titulo', 'ID': 'Id'})
        .copy()
    )

    return final_df


def join_interaction(row, description_column, comment_column):

    columns_to_join = [description_column] + [
        col for col in row.index if col.startswith(comment_column)
    ]

    values = []
    for column in columns_to_join:
        value = row[column]
        if pd.notnull(value):
            value = value.strip()
            if value:
                value = normalize_text(value, patterns)
                if value != '':
                    values.append(value)

    return ' | '.join(values)

def parse_datetime_string_jira(date):
    months = {
        'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4,
        'mai': 5, 'jun': 6, 'jul': 7, 'ago': 8,
        'set': 9, 'out': 10, 'nov': 11, 'dez': 12
    }

    date_part, time_part, period = date.split()
    day_str, month_str, year_short = date_part.split('/')
    day = int(day_str)
    month = months[month_str.lower()]
    year = 2000 + int(year_short)

    hour_str, minute_str = time_part.split(':')
    hour = int(hour_str)
    minute = int(minute_str)
    period = period.upper()

    if period == 'PM' and hour != 12:
        hour += 12
    elif period == 'AM' and hour == 12:
        hour = 0

    return datetime(year, month, day, hour, minute)

def parse_datetime_string_porto(date):
    date_part, time_part = date.split()
    day_str, month_str, year_short = date_part.split('/')
    day = int(day_str)
    month = int(month_str)
    year = int(year_short)

    hour_str, minute_str = time_part.split(':')
    hour = int(hour_str)
    minute = int(minute_str)

    return datetime(year, month, day, hour, minute)

def safe_format_jira(date_str):
    if pd.isna(date_str) or not str(date_str).strip():
        return ""
    dt = parse_datetime_string_jira(date_str)

    return dt.isoformat()

def safe_format_porto(date_str):
    if pd.isna(date_str) or not str(date_str).strip():
        return ""
    dt = parse_datetime_string_porto(date_str)

    return dt.isoformat()
