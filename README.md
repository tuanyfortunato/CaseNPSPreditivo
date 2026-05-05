# Case NPS Preditivo

## Objetivo do projeto
Analisar dados operacionais de pedidos, logística e atendimento para identificar fatores que impactam o NPS e construir um modelo preditivo capaz de antecipar clientes detratores.

## Descrição da base de dados
O dataset contém informações por pedido, incluindo:
- dados do pedido: valor, quantidade de itens, desconto, forma de pagamento
- dados logísticos: tempo de entrega, atraso, tentativas, frete
- atendimento: contatos com SAC, tempo de resolução, reclamações
- indicadores internos: recompra em 30 dias, score interno de satisfação e NPS

## Metodologia utilizada
- Entendimento do negócio e definição do alvo
- Limpeza e engenharia de features
- Análise exploratória (correlação, gráficos, teste de hipótese)
- Modelo de classificação para prever detratores
- Avaliação de performance com acurácia, ROC AUC, relatório de classificação e matriz de confusão

## Como reproduzir os resultados
1. Ative o ambiente virtual: `venv\Scripts\Activate.ps1`
2. Instale dependências: `pip install -r requirements.txt`
3. Execute o script principal: `python src\analise_nps.py`
4. Abra o notebook: `notebooks\CaseNpsPreditivo_Final.ipynb`

## Estrutura do repositório
- `data/`: base de dados CSV
- `notebooks/`: análise exploratória e storytelling
- `src/`: script de análise e modelagem
- `reports/`: espaço para entregáveis adicionais

