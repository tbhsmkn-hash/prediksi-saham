import streamlit as st
import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import datetime

# --- Pengaturan Halaman Utama ---
st.title("Aplikasi Prediksi Harga Cryptocurrency-ARIMA")
st.write("Aplikasi ini melatih model ARIMA secara real-time menggunakan data historis langsung dari Yahoo Finance.")

# --- Pengaturan Data di Sidebar ---
st.sidebar.header("1. Pengaturan Data Crypto")
# Menggunakan ticker Crypto (format yfinance untuk crypto umumnya diakhiri dengan '-USD')
# --- DAFTAR 50 COIN CRYPTO TERPOPULER (Untuk Dropdown) ---
CRYPTO_LIST = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", 
    "ADA-USD", "DOGE-USD", "SHIB-USD", "AVAX-USD", "DOT-USD",
    "MATIC-USD", "LINK-USD", "TRX-USD", "UNI-USD", "ICP-USD",
    "BCH-USD", "NEAR-USD", "FIL-USD", "LTC-USD", "LEO-USD",
    "ETC-USD", "APT-USD", "ATOM-USD", "XLM-USD", "HBAR-USD",
    "IMX-USD", "GRT-USD", "STX-USD", "KAS-USD", "TON-USD",
    "PEPE-USD", "WIF-USD", "FLOKI-USD", "RUNE-USD", "RENDER-USD",
    "THETA-USD", "FTM-USD", "LUNC-USD", "AAVE-USD", "ALGO-USD",
    "EGLD-USD", "FLOW-USD", "MKR-USD", "SAND-USD", "MANA-USD",
    "XTZ-USD", "VET-USD", "AXS-USD", "EOS-USD", "NEO-USD"
]

# --- Pengaturan Data di Sidebar (Ubah ke Dropdown Selectbox) ---
st.sidebar.header("1. Pengaturan Data Crypto")

# Pengguna sekarang memilih lewat dropdown, tidak perlu mengetik lagi
selected_crypto = st.sidebar.selectbox("Pilih Koin Crypto", CRYPTO_LIST)

selected_start_date = st.sidebar.date_input("Tanggal Mulai", datetime.date(2021, 1, 1))
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
def get_crypto_data(ticker, start, end):
    """Mengambil data historis cryptocurrency menggunakan pustaka yfinance."""
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
df_crypto = get_crypto_data(selected_crypto, start_str, end_str)

# --- Alur Tampilan Utama Aplikasi ---
if not df_crypto.empty and 'Close' in df_crypto.columns:
    st.subheader(f"Data Historis Harga {selected_crypto}")
    st.write(df_crypto[['Open', 'High', 'Low', 'Close', 'Volume']].tail()) 
    
    # Visualisasi Data Historis
    st.line_chart(df_crypto['Close']) 
    
    # --- Tombol Eksekusi Model ARIMA ---
    if st.button("Jalankan Prediksi ARIMA"):
        with st.spinner(f"Melatih model ARIMA untuk {selected_crypto}..."):
            try:
                # PERBAIKAN UTAMA: Crypto berjalan 24/7. 
                # Kita gunakan freq='D' (Daily/Kalender murni) dan menghapus dropna/ffill bursa saham.
                series_close = df_crypto['Close'].asfreq('D')
                
                # Jika ada lubang data langka, tetap diisi agar model tidak patah
                if series_close.isnull().any():
                    series_close = series_close.ffill()
                
                # Memasang dan melatih model ARIMA
                model = ARIMA(series_close, order=(p, d, q))
                model_fit = model.fit()
                
                # 2. Peramalan (Forecast)
                forecast = model_fit.forecast(steps=forecast_steps)
                
                # 3. Membuat Indeks Tanggal Hasil Prediksi (Setiap hari berturut-turut tanpa lompat akhir pekan)
                last_date = series_close.index[-1]
                forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
                df_forecast = pd.DataFrame({'Prediksi Harga ($)': forecast.values}, index=forecast_index)
                df_forecast.index.name = 'Tanggal'
                
                # Tampilkan Tabel Hasil Peramalan
                st.success(f"Proses peramalan selesai untuk {forecast_steps} hari ke depan!")
                st.write(df_forecast)
                
                # 4. Plot Visualisasi Gabungan
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(series_close.tail(40), label='Data Historis (40 Hari Terakhir)', color='orange', linewidth=2)
                ax.plot(df_forecast['Prediksi Harga ($)'], label='Hasil Prediksi ARIMA', color='blue', linestyle='--', marker='o')
                ax.set_title(f"Grafik Estimasi Tren Harga {selected_crypto}", fontsize=14)
                ax.set_xlabel("Tanggal", fontsize=11)
                ax.set_ylabel("Harga (USD)", fontsize=11)
                ax.legend()
                ax.grid(True, linestyle=':', alpha=0.6)
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Gagal memproses model ARIMA. Masalah: {e}")
else:
    st.warning(f"Data untuk simbol '{selected_crypto}' tidak ditemukan. Tip: Gunakan akhiran '-USD' (contoh: BTC-USD, ETH-USD, DOGE-USD).")
