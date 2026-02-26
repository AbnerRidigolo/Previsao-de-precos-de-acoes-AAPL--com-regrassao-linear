import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_visualizations():
    # Carregar resultados de teste
    test_results = pd.read_csv("test_results.csv", index_col="Date", parse_dates=True)
    
    # 1. Gráfico de Preços Reais vs Previstos
    plt.figure(figsize=(14, 7))
    plt.plot(test_results.index, test_results['Actual'], label='Preço Real', color='blue', alpha=0.7)
    plt.plot(test_results.index, test_results['Predicted'], label='Preço Previsto', color='red', linestyle='--', alpha=0.7)
    plt.title('AAPL: Preço Real vs Previsto (Conjunto de Teste)', fontsize=16)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Preço de Fechamento (USD)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("aapl_prediction_plot.png")
    plt.close()
    
    # 2. Gráfico de Importância dos Atributos (Coeficientes)
    importance = pd.read_csv("feature_importance.csv")
    importance = importance.sort_values(by='Coefficient', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Feature', data=importance, palette='viridis')
    plt.title('Importância dos Atributos (Coeficientes da Regressão Linear)', fontsize=14)
    plt.xlabel('Valor do Coeficiente', fontsize=12)
    plt.ylabel('Atributo', fontsize=12)
    plt.tight_layout()
    plt.savefig("feature_importance_plot.png")
    plt.close()
    
    # 3. Gráfico de Resíduos
    test_results['Residuals'] = test_results['Actual'] - test_results['Predicted']
    plt.figure(figsize=(10, 6))
    sns.histplot(test_results['Residuals'], kde=True, color='purple')
    plt.title('Distribuição dos Resíduos (Erros de Previsão)', fontsize=14)
    plt.xlabel('Erro (Real - Previsto)', fontsize=12)
    plt.ylabel('Frequência', fontsize=12)
    plt.savefig("residuals_distribution.png")
    plt.close()
    
    print("Visualizações geradas com sucesso.")

if __name__ == "__main__":
    generate_visualizations()