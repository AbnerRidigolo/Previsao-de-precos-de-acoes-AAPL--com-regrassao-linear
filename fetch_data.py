import yfinance as yf
import pandas as pd

def fetch_aapl_data():
    # Baixar dados históricos da AAPL (últimos 5 anos)
    aapl = yf.Ticker("AAPL")
    df = aapl.history(period="5y")
    
    # Salvar em CSV para as próximas fases
    df.to_csv("aapl_historical_data.csv")
    print(f"Dados baixados com sucesso. Total de registros: {len(df)}")
    print(df.head())

if __name__ == "__main__":
    fetch_aapl_data()