"""Análise do desafio de NPS.

Este script traz o caminho completo: leitura dos dados, tratamento,
algumas análises e teste de hipótese para entender o impacto do atraso
no NPS.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_FILENAME = 'desafio_nps_fase_1.csv'

# Ponto de entrada do script
def main() -> None:
    # carrega os dados
    df = load_data()

    # prepara o conjunto de variáveis
    df = preprocess(df)

    # mostra um resumo rápido da base
    data_overview(df)

    # calcula correlação com o NPS
    correlation_analysis(df)
    
    # compara NPS de pedidos no prazo versus atrasados
    hypothesis_test(df)


# Carrega o arquivo CSV
def load_data(data_path: str | None = None) -> pd.DataFrame:
    if data_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, '..', 'data', DATA_FILENAME)
    return pd.read_csv(data_path)

# Limpa dados e cria variáveis de NPS
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=['customer_id', 'order_id'])

    df['nps_category'] = pd.cut(
        df['nps_score'],
        bins=[-0.1, 6, 8, 10],
        labels=['Detrator', 'Passivo', 'Promotor'],
    )

    df['is_detractor'] = (df['nps_score'] <= 6).astype(int)
    return df


# Mostra os principais números da base
def data_overview(df: pd.DataFrame) -> None:
    print('--- Visão geral dos dados ---')
    print('Shape:', df.shape)
    print('\nValores ausentes:')
    print(df.isna().sum())
    print('\nContagem por categoria de NPS:')
    print(df['nps_category'].value_counts())
    print('\n')

# Calcula correlação entre as variáveis numéricas e o NPS
def correlation_analysis(df: pd.DataFrame) -> pd.Series:
    numeric = df.select_dtypes(include=[np.number])
    correlation = numeric.corr()['nps_score'].sort_values(ascending=False)
    print('--- Correlação com NPS Score ---')
    print(correlation)
    return correlation

# Compara NPS de pedidos no prazo e atrasados
def hypothesis_test(df: pd.DataFrame) -> None:
    atrasos = df[df['delivery_delay_days'] > 0]['nps_score']
    prazo = df[df['delivery_delay_days'] == 0]['nps_score']
    t_stat, p_val = stats.ttest_ind(atrasos, prazo, equal_var=False)

    print('--- Teste de Hipótese: Atraso vs Prazo ---')
    print(f'Estatística T: {t_stat:.4f}')
    print(f'P-valor: {p_val:.4e}')
    if p_val < 0.05:
        print('Resultado: diferença estatisticamente significativa.')
    else:
        print('Resultado: não há evidência de diferença significativa.')
    print('\n')

# Execução do script
if __name__ == '__main__':
    main()
