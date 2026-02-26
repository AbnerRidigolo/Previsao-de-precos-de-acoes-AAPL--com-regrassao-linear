# Previsão de Preços de Ações (AAPL) com Regressão Linear

Este projeto utiliza técnicas de **Ciência de Dados** e **Machine Learning** para prever o preço de fechamento das ações da Apple (AAPL) para o próximo dia, utilizando dados históricos do Yahoo Finance.

##  Sobre o Projeto

O objetivo é construir um pipeline completo de análise preditiva, desde a coleta de dados brutos até a visualização dos resultados. O modelo utiliza **Regressão Linear** com engenharia de atributos (features) para identificar padrões nos preços históricos.

###  Tecnologias Utilizadas

*   **Python 3.11+**
*   **yfinance**: Coleta de dados financeiros em tempo real.
*   **Scikit-learn**: Treinamento e avaliação do modelo de Machine Learning.
*   **Pandas & Numpy**: Manipulação e processamento de dados.
*   **Matplotlib & Seaborn**: Geração de gráficos e visualizações.

---

##  Estrutura do Projeto

O projeto é dividido em quatro scripts principais que devem ser executados em ordem:

1.  `fetch_data.py`: Baixa os últimos 5 anos de dados históricos da AAPL.
2.  `feature_engineering.py`: Cria indicadores técnicos (Médias Móveis, RSI, Volatilidade).
3.  `train_model.py`: Treina o modelo de Regressão Linear e faz a previsão para o próximo dia.
4.  `visualize_results.py`: Gera gráficos de desempenho e análise de erros.

---

##  Como Executar

### 1. Instale as dependências
No seu terminal, execute:
```bash
python -m pip install yfinance scikit-learn pandas numpy matplotlib seaborn joblib
```

### 2. Execute o Pipeline
Rode os scripts na ordem abaixo:
```bash
python fetch_data.py
python feature_engineering.py
python train_model.py
python visualize_results.py
```

---

##  Entendendo os Arquivos Gerados

Após a execução, os seguintes arquivos serão criados na sua pasta:

| Arquivo | Tipo | Descrição |
| :--- | :--- | :--- |
| `aapl_historical_data.csv` | Dados | Preços brutos baixados do Yahoo Finance. |
| `aapl_features.csv` | Dados | Dados processados com indicadores técnicos (SMA, RSI). |
| `aapl_model.pkl` | Modelo | O "cérebro" do modelo treinado para uso futuro. |
| `test_results.csv` | Resultados | Comparação entre Preço Real vs. Preço Previsto. |
| `aapl_prediction_plot.png` | Gráfico | Visualização da performance do modelo no tempo. |
| `feature_importance_plot.png` | Gráfico | Quais fatores mais influenciaram a previsão. |
| `residuals_distribution.png` | Gráfico | Análise estatística dos erros do modelo. |

---

##  Resultados Obtidos

O modelo demonstrou uma alta correlação com os preços reais, atingindo um **R² Score de aproximadamente 0.97**. 

> **Nota:** Este projeto tem fins educacionais e não deve ser utilizado como recomendação de investimento. O mercado financeiro é volátil e modelos lineares simples possuem limitações.

---
Desenvolvido por [Seu Nome] com auxílio
