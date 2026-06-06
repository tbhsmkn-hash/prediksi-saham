import streamlit as st
import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import datetime

# --- Pengaturan Halaman Utama ---
st.title("Aplikasi Prediksi Harga Saham Berbasis Model ARIMA")

# --- Pengaturan Data di Sidebar ---
st.sidebar.header("1. Pengaturan Data Saham")
selected_ticker = st.sidebar.text_input("Simbol Saham (Ticker)", "AAPL").upper()
selected_start_date = st.sidebar.date_input("Tanggal Mulai", datetime.date(2020, 1, 1))
selected_end_date = st.sidebar.date_input("Tanggal Akhir", datetime.date.today())

if selected_end_date < selected_start_date:
    st.sidebar.error("Kesalahan: Tanggal akhir harus setelah tanggal mulai.")
    st.stop()

# --- Pengaturan Parameter ARIMA di Sidebar ---
st.sidebar.header("2. Parameter Model ARIMA")
p = st.sidebar.number_input("Orde AR (p)", min_value=0, max_value=5, value=1)
d = st.sidebar.number_input("Differencing (d)", min_value=0, max_value=2, value=1)
q = st.sidebar.number_input("Orde MA (q)", min_value=0, max_value=5, value=1)

forecast_steps = st.sidebar.slider("Horizon Prediksi (Hari ke Depan)", 1, 30, 7)

# --- Fungsi Mengambil Data dari Yahoo Finance ---
@st.cache_data 
def get_stock_data(ticker, start, end):
    """Mengambil data historis saham menggunakan pustaka yfinance."""
    try:
        df = yf.download(ticker, start=start, end=end)
        if not df.empty:
            # Mengatasi MultiIndex kolom jika terjadi pada library yfinance versi terbaru
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            # Memastikan index berupa datetime
            df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.error(f"Terjadi kesalahan saat mengunduh data dari Yahoo Finance: {e}")
        return pd.DataFrame()

# --- Eksekusi Pengambilan Data ---
start_str = selected_start_date.strftime("%Y-%m-%d")
end_str = selected_end_date.strftime("%Y-%m-%d")
df_stock = get_stock_data(selected_ticker, start_str, end_str)

# --- Alur Tampilan Utama Aplikasi ---
if not df_stock.empty and 'Close' in df_stock.columns:
    st.subheader(f"Data Historis Harga Saham {selected_ticker}")
    st.write(df_stock[['Open', 'High', 'Low', 'Close', 'Volume']].tail()) 
    
    # Visualisasi Data Historis
    st.line_chart(df_stock['Close']) 
    
    # --- Tombol Eksekusi Model ARIMA ---
    if st.button("Jalankan Prediksi ARIMA"):
        with st.spinner("Melatih model ARIMA dan menghitung ramalan harga..."):
            try:
                # 1. Training Model ARIMA menggunakan seri Close price
                # Memastikan frekuensi index diatur (b/business day karena bursa libur di akhir pekan)
                series_close = df_stock['Close'].asfreq('B').fillna(method='ffill')
                
                model = ARIMA(series_close, order=(p, d, q))
                model_fit = model.fit()
                
                # 2. Peramalan (Forecast)
                forecast = model_fit.forecast(steps=forecast_steps)
                
                # 3. Membuat Indeks Tanggal Hasil Prediksi (Hanya hari kerja/Business Days)
                last_date = series_close.index[-1]
                forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='B')
                df_forecast = pd.DataFrame({'Prediksi Harga ($)': forecast.values}, index=forecast_index)
                df_forecast.index.name = 'Tanggal'
                
                # Tampilkan Tabel Hasil Peramalan
                st.success(f"Proses peramalan selesai untuk {forecast_steps} hari kerja ke depan!")
                st.write(df_forecast)
                
                # 4. Plot Visualisasi Gabungan (Data Historis Terakhir & Hasil Prediksi)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(series_close.tail(40), label='Data Historis (40 Hari Terakhir)', color='blue', linewidth=2)
                ax.plot(df_forecast['Prediksi Harga ($)'], label='Hasil Prediksi ARIMA', color='red', linestyle='--', marker='o')
                ax.set_title(f"Grafik Estimasi Tren Harga Saham {selected_ticker}", fontsize=14)
                ax.set_xlabel("Tanggal", fontsize=11)
                ax.set_ylabel("Harga Penutupan", fontsize=11)
                ax.legend()
                ax.grid(True, linestyle=':', alpha=0.6)
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Gagal memproses model ARIMA. Pastikan kombinasi parameter (p,d,q) sesuai dengan karakteristik data. Detail Error: {e}")
else:
    st.warning(f"Data untuk simbol '{selected_ticker}' tidak ditemukan atau format kolom 'Close' tidak sesuai. Silakan periksa kembali penulisan simbol emiten Anda.")
