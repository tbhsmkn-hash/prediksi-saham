import streamlit as st
import yfinance as yf
import pandas as pd
import datetime

# --- Pilih Saham & Periode ---
TICKER_SYMBOL = "AAPL" 
START_DATE = "2020-01-01" 
END_DATE = datetime.date.today().strftime("%Y-%m-%d") 

st.title(f"Prediksi Harga Saham {TICKER_SYMBOL}")

@st.cache_data 
def get_stock_data(ticker, start, end):
    """Mengambil data historis saham dari Yahoo Finance."""
    try:
        # Perbaikan KeyError: Tambahkan auto_adjust=True agar kolom menjadi Single Index yang bersih
        data = yf.download(ticker, start=start, end=end, auto_adjust=True)
        data.dropna(inplace=True)
        return data
    except Exception as e:
        st.error(f"Gagal mengambil data saham untuk {ticker}: {e}")
        return pd.DataFrame()

# --- Tampilkan Input Pengguna untuk Saham dan Tanggal ---
st.sidebar.header("Pengaturan Data")
selected_ticker = st.sidebar.text_input("Simbol Saham", TICKER_SYMBOL).upper()
selected_start_date = st.sidebar.date_input("Tanggal Mulai", datetime.date(2020, 1, 1))
selected_end_date = st.sidebar.date_input("Tanggal Akhir", datetime.date.today())

if selected_end_date < selected_start_date:
    st.sidebar.error("Tanggal akhir harus setelah tanggal mulai.")
    st.stop() 

# --- Ambil Data ---
df_stock = get_stock_data(selected_ticker, selected_start_date.strftime("%Y-%m-%d"), selected_end_date.strftime("%Y-%m-%d"))

if not df_stock.empty:
    st.subheader(f"Data Historis Harga Saham {selected_ticker}")
    st.write(df_stock.tail()) 
    
    # Perbaikan KeyError: Karena auto_adjust=True, gunakan kolom 'Close'
    if 'Close' in df_stock.columns:
        st.line_chart(df_stock['Close']) 
    else:
        # Cadangan jika kolom MultiIndex tetap terbentuk di beberapa versi enviroment
        st.line_chart(df_stock.iloc[:, 0]) 
else:
    st.warning("Tidak ada data saham yang tersedia atau terjadi kesalahan saat mengambil data.")
