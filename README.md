# Case NPS Preditivo

Projeto para analisar dados de pedidos e identificar quais fatores operacionais têm mais impacto no NPS.

## Metodologia utilizada
- Entendimento do negócio e definição do alvo
- Limpeza e engenharia de features
- Análise exploratória (correlação, gráficos, teste de hipótese)
- Modelo de classificação para prever detratores
- Avaliação de performance com acurácia, ROC AUC, relatório de classificação e matriz de confusão

## Dados usados
A base traz informações de cada pedido, incluindo:
- detalhes do pedido: valor, quantidade, desconto e forma de pagamento
- logística: prazo, atraso, número de tentativas e frete
- atendimento: contatos com SAC, reclamações e tempo de resolução
- indicadores internos: NPS e score interno de satisfação

## Como rodar
1. Ative o ambiente virtual: `venv\Scripts\Activate.ps1`
2. Instale as dependências: `pip install -r requirements.txt`
3. Execute o script: `python src\analise_nps.py`
4. Veja o notebook: `notebooks\CaseNpsPreditivo_Final.ipynb`

## Estrutura
- `data/`: arquivo CSV do desafio
- `src/`: código principal de análise
- `notebooks/`: notebook com a jornada de análise
- `reports/`: material de apoio ou apresentações

