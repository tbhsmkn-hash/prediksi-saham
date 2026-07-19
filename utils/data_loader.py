import yfinance as yf
import streamlit as st

@st.cache_data(ttl=3600)
def get_crypto_data(ticker, period="365d"):
    try:
        # Menarik data dari Yahoo Finance
        df = yf.download(ticker, period=period, interval="1d")
        
        # yfinance terkadang mengembalikan multi-index, pastikan dataframe bersih
        if not df.empty:
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            return df
        else:
            return None
    except Exception as e:
        st.error(f"Error saat mengambil data: {e}")
        return None