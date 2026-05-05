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


def load_data(data_path: str | None = None) -> pd.DataFrame:
    if data_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, '..', 'data', DATA_FILENAME)
    return pd.read_csv(data_path)


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


def data_overview(df: pd.DataFrame) -> None:
    print('--- Data Overview ---')
    print('Shape:', df.shape)
    print('\nMissing values:')
    print(df.isna().sum())
    print('\nNPS category counts:')
    print(df['nps_category'].value_counts())
    print('\n')


def correlation_analysis(df: pd.DataFrame) -> pd.Series:
    numeric = df.select_dtypes(include=[np.number])
    correlation = numeric.corr()['nps_score'].sort_values(ascending=False)
    print('--- Correlação com NPS Score ---')
    print(correlation)
    return correlation


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


def build_classification_pipeline(df: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    features = df.drop(columns=['nps_score', 'nps_category', 'is_detractor'])
    target = df['is_detractor']

    numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = features.columns.difference(numeric_features).tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features),
        ],
        remainder='passthrough',
    )

    model = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model, X_train, y_train, X_test, y_test


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    print('--- Avaliação do Modelo de Classificação ---')
    print('Acurácia:', accuracy_score(y_test, predictions))
    print('ROC AUC:', roc_auc_score(y_test, probabilities))
    print('\nClassification report:')
    print(classification_report(y_test, predictions, target_names=['Não Detrator', 'Detrator']))
    print('Confusion matrix:')
    print(confusion_matrix(y_test, predictions))


def get_feature_importance(model: Pipeline, X_train: pd.DataFrame) -> pd.Series | None:
    classifier = model.named_steps['classifier']
    preprocessor = model.named_steps['preprocessor']
    if hasattr(classifier, 'coef_'):
        numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X_train.columns.difference(numeric_features).tolist()
        cat_encoder = preprocessor.named_transformers_['cat']
        cat_features = []
        if hasattr(cat_encoder, 'get_feature_names_out'):
            cat_features = cat_encoder.get_feature_names_out(categorical_features).tolist()
        feature_names = numeric_features + cat_features
        coefficients = classifier.coef_[0]
        importance = pd.Series(coefficients, index=feature_names).sort_values()
        return importance
    return None


def main() -> None:
    df = load_data()
    df = preprocess(df)
    data_overview(df)
    correlation_analysis(df)
    hypothesis_test(df)

    model, X_train, y_train, X_test, y_test = build_classification_pipeline(df)
    evaluate_model(model, X_test, y_test)

    importance = get_feature_importance(model, X_train)
    if importance is not None:
        print('\n--- Importância das Variáveis (coeficientes do modelo) ---')
        print(importance)


if __name__ == '__main__':
    main()
