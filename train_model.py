import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

def train_and_evaluate():
    df = pd.read_csv("aapl_features.csv", index_col="Date", parse_dates=True)
    
    # Selecionar atributos (X) e alvo (y)
    features = ['Close', 'SMA_5', 'SMA_20', 'SMA_50', 'Volatility_20', 'RSI_14', 'Daily_Return', 'Lag_1', 'Lag_2', 'Lag_3']
    X = df[features]
    y = df['Target']
    
    # Dividir em treino e teste (80/20) - Mantendo a ordem temporal
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Treinar o modelo
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Fazer previsões
    y_pred = model.predict(X_test)
    
    # Avaliar o modelo
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("--- Métricas do Modelo ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Salvar o modelo e os dados de teste para visualização
    joblib.dump(model, "aapl_model.pkl")
    test_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=y_test.index)
    test_results.to_csv("test_results.csv")
    
    # Previsão para o próximo dia (usando os dados mais recentes disponíveis)
    latest_data = X.iloc[-1:].values
    next_day_pred = model.predict(latest_data)[0]
    
    # Pegar o último preço real disponível (que seria o alvo do último registro se tivéssemos)
    # Na verdade, o 'Target' do último registro no CSV original era o preço do dia seguinte.
    # Vamos carregar os dados originais para comparar com o preço real mais recente.
    original_df = pd.read_csv("aapl_historical_data.csv", index_col="Date", parse_dates=True)
    last_actual_price = original_df['Close'].iloc[-1]
    
    print("\n--- Previsão ---")
    print(f"Último preço de fechamento conhecido: {last_actual_price:.2f}")
    print(f"Preço previsto para o próximo dia: {next_day_pred:.2f}")
    
    # Salvar importância dos atributos
    importance = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_})
    importance.to_csv("feature_importance.csv", index=False)

if __name__ == "__main__":
    train_and_evaluate()