import pandas as pd
import numpy as np

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def engineer_features():
    df = pd.read_csv("aapl_historical_data.csv", index_col="Date", parse_dates=True)
    
    # Médias Móveis (SMA)
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Volatilidade (Desvio Padrão de 20 dias)
    df['Volatility_20'] = df['Close'].rolling(window=20).std()
    
    # RSI (Relative Strength Index)
    df['RSI_14'] = calculate_rsi(df['Close'])
    
    # Retornos diários e defasados (Lags)
    df['Daily_Return'] = df['Close'].pct_change()
    df['Lag_1'] = df['Close'].shift(1)
    df['Lag_2'] = df['Close'].shift(2)
    df['Lag_3'] = df['Close'].shift(3)
    
    # Variável Alvo: Preço de Fechamento do Próximo Dia
    df['Target'] = df['Close'].shift(-1)
    
    # Remover valores nulos gerados pelas janelas móveis e lags
    df_cleaned = df.dropna()
    
    df_cleaned.to_csv("aapl_features.csv")
    print(f"Engenharia de atributos concluída. Registros após limpeza: {len(df_cleaned)}")
    print(df_cleaned[['Close', 'SMA_5', 'RSI_14', 'Target']].tail())

if __name__ == "__main__":
    engineer_features()