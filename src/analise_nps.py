"""Análise de NPS Preditivo.

Este script carrega a base de dados de pedidos e interações com o cliente,
realiza limpeza, engenharia de features e treina um modelo de classificação
para prever clientes detratores antes da pesquisa de NPS ser aplicada.

O alvo é `is_detractor`, derivado de `nps_score <= 6`.
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

# Função principal para executar a análise completa de NPS preditivo, desde o carregamento dos dados até a avaliação do modelo e extra
def main() -> None:
    #Carregamento dos dados
    df = load_data()

    #Pré-Processamento
    df = preprocess(df)

    # Análise exploratória
    data_overview(df)

    # Análise de correlação
    correlation_analysis(df)
    
    # Teste de hipótese
    hypothesis_test(df)


# Buscando os dados e pré-processamento
def load_data(data_path: str | None = None) -> pd.DataFrame:
    if data_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, '..', 'data', DATA_FILENAME)
    return pd.read_csv(data_path)

#Limpeza do Dataframe e classificando os clientes em detratores, passivos e promotores
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


# Análise exploratória
def data_overview(df: pd.DataFrame) -> None:
    print('--- Visão geral dos dados ---')
    print('Shape:', df.shape)
    print('\nValores ausentes:')
    print(df.isna().sum())
    print('\nContagem por categoria de NPS:')
    print(df['nps_category'].value_counts())
    print('\n')

# Correlação entre variáveis numéricas e NPS Score
def correlation_analysis(df: pd.DataFrame) -> pd.Series:
    numeric = df.select_dtypes(include=[np.number])
    correlation = numeric.corr()['nps_score'].sort_values(ascending=False)
    print('--- Correlação com NPS Score ---')
    print(correlation)
    return correlation

# Teste de hipótese para comparar NPS Score entre clientes com atraso e sem atraso
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
