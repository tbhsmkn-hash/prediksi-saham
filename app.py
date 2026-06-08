import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import datetime

# --- Pengaturan Halaman Utama ---
st.title("Aplikasi Prediksi Cryptocurrency Berbasis Hybrid ARIMA-SVR")
st.write("Aplikasi ini mengombinasikan model klasik ARIMA (Linear) dan Machine Learning SVR (Non-Linear) secara real-time.")

# --- DAFTAR 50 COIN CRYPTO TERPOPULER (Untuk Dropdown sesuai struktur awal) ---
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

# --- Pengaturan Data di Sidebar ---
st.sidebar.header("1. Pengaturan Data Crypto")
selected_crypto = st.sidebar.selectbox("Pilih Koin Crypto", CRYPTO_LIST)
selected_start_date = st.sidebar.date_input("Tanggal Mulai", datetime.date(2021, 1, 1))
selected_end_date = st.sidebar.date_input("Tanggal Akhir", datetime.date.today())

if selected_end_date < selected_start_date:
    st.sidebar.error("Kesalahan: Tanggal akhir harus setelah tanggal mulai.")
    st.stop()

# --- Pengaturan Parameter Model di Sidebar ---
st.sidebar.header("2. Parameter Model Hybrid")
st.sidebar.subheader("Parameter ARIMA (Linear)")
p = st.sidebar.number_input("Orde AR (p)", min_value=0, max_value=5, value=1)
d = st.sidebar.number_input("Differencing (d)", min_value=0, max_value=2, value=1)
q = st.sidebar.number_input("Orde MA (q)", min_value=0, max_value=5, value=1)

st.sidebar.subheader("Parameter SVR (Non-Linear)")
svr_kernel = st.sidebar.selectbox("Kernel SVR", ["rbf", "linear", "poly"])
svr_c = st.sidebar.number_input("Nilai Regulasi (C)", min_value=0.1, max_value=100.0, value=10.0, step=1.0)

forecast_steps = st.sidebar.slider("Horizon Prediksi (Hari ke Depan)", 1, 30, 7)

# --- Fungsi Mengambil Data dari Yahoo Finance ---
@st.cache_data 
def get_crypto_data(ticker, start, end):
    """Mengambil data historis cryptocurrency menggunakan pustaka yfinance."""
    try:
        df = yf.download(ticker, start=start, end=end)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.error(f"Terjadi kesalahan saat mengunduh data: {e}")
        return pd.DataFrame()

# --- Eksekusi Pengambilan Data ---
start_str = selected_start_date.strftime("%Y-%m-%d")
end_str = selected_end_date.strftime("%Y-%m-%d")
df_crypto = get_crypto_data(selected_crypto, start_str, end_str)

# --- Alur Tampilan Utama Aplikasi ---
if not df_crypto.empty and 'Close' in df_crypto.columns:
    st.subheader(f"Data Historis Harga {selected_crypto}")
    st.write(df_crypto[['Open', 'High', 'Low', 'Close', 'Volume']].tail()) 
    st.line_chart(df_crypto['Close']) 
    
    # --- Tombol Eksekusi Model Hybrid ---
    if st.button("Jalankan Prediksi Hybrid ARIMA-SVR"):
        with st.spinner(f"Melatih model Hybrid untuk {selected_crypto}..."):
            try:
                # Memastikan frekuensi data harian penuh (24/7)
                series_close = df_crypto['Close'].asfreq('D')
                if series_close.isnull().any():
                    series_close = series_close.ffill()
                
                # ==========================================
                # LENGKAH 1: TRAINING MODEL ARIMA (LINEAR)
                # ==========================================
                model_arima = ARIMA(series_close, order=(p, d, q))
                model_arima_fit = model_arima.fit()
                
                # Mendapatkan nilai fitted values (prediksi data historis oleh ARIMA)
                arima_fitted = model_arima_fit.fittedvalues
                
                # Hitung nilai Residual (Error non-linear)
                residuals = series_close - arima_fitted
                
                # ==========================================
                # LANGKAH 2: TRAINING MODEL SVR (NON-LINEAR)
                # ==========================================
                # Menyiapkan fitur lag (misal menggunakan 3 nilai residual terakhir untuk memprediksi residual esok)
                look_back = 3
                X_svr, y_svr = [], []
                res_values = residuals.values
                
                for i in range(len(res_values) - look_back):
                    X_svr.append(res_values[i:(i + look_back)])
                    y_svr.append(res_values[i + look_back])
                
                X_svr, y_svr = np.array(X_svr), np.array(y_svr)
                
                # Normalisasi Fitur untuk SVR agar hasil optimal
                scaler = StandardScaler()
                X_svr_scaled = scaler.fit_transform(X_svr)
                
                # Melatih model SVR
                model_svr = SVR(kernel=svr_kernel, C=svr_c)
                model_svr.fit(X_svr_scaled, y_svr)
                
                # ==========================================
                # LANGKAH 3: PROSES PERAMALAN (FORECASTING)
                # ==========================================
                # 3a. Peramalan nilai Linear dengan ARIMA
                arima_forecast = model_arima_fit.forecast(steps=forecast_steps)
                
                # 3b. Peramalan nilai Non-Linear dengan SVR secara berantai (Iterative)
                svr_forecast = []
                input_res = list(res_values[-look_back:]) # Ambil data residual terakhir
                
                for _ in range(forecast_steps):
                    input_res_scaled = scaler.transform([input_res[-look_back:]])
                    pred_res = model_svr.predict(input_res_scaled)[0]
                    svr_forecast.append(pred_res)
                    input_res.append(pred_res) # Memasukkan hasil prediksi sebagai input berikutnya
                
                # 3c. Menggabungkan Hasil Prediksi (ARIMA + SVR)
                hybrid_forecast = arima_forecast.values + np.array(svr_forecast)
                
                # ==========================================
                # LANGKAH 4: VISUALISASI DAN OUTPUT
                # ==========================================
                # Membuat Indeks Tanggal Masa Depan
                last_date = series_close.index[-1]
                forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
                
                # Membuat Dataframe Hasil
                df_forecast = pd.DataFrame({
                    'Prediksi ARIMA (Linear)': arima_forecast.values,
                    'Koreksi SVR (Non-Linear)': svr_forecast,
                    'Total Prediksi Hybrid ($)': hybrid_forecast
                }, index=forecast_index)
                df_forecast.index.name = 'Tanggal'
                
                # ... (Kode Langkah 4: Tampilkan Tabel Hasil Peramalan sebelumnya) ...
                st.success(f"Proses peramalan Hybrid selesai untuk {forecast_steps} hari ke depan!")
                st.write(df_forecast)
                
                # ==========================================
                # FITUR TAMBAHAN: DOWNLOAD DATASET TO .CSV
                # ==========================================
                # Mengonversi DataFrame ke format CSV (dengan encoding utf-8)
                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=True).encode('utf-8')
                
                csv_data = convert_df_to_csv(df_forecast)
                
                # Menampilkan tombol unduh tepat di bawah tabel
                st.download_button(
                    label="📥 Unduh Hasil Prediksi (.CSV)",
                    data=csv_data,
                    file_name=f"prediksi_hybrid_{selected_crypto}_{last_date.strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="download-csv"
                )
                
                # ... (Kode Plot Visualisasi Grafis selanjutnya) ...
                # Plot Visualisasi Perbandingan Grafis
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(series_close.tail(30), label='Data Historis Aktual (30 Hari Terakhir)', color='orange', linewidth=2)
                ax.plot(df_forecast['Total Prediksi Hybrid ($)'], label='Hasil Akhir Hybrid (ARIMA+SVR)', color='green', linestyle='--', marker='o')
                ax.plot(df_forecast['Prediksi ARIMA (Linear)'], label='Prediksi Standar ARIMA saja', color='red', linestyle=':', alpha=0.7)
                
                ax.set_title(f"Grafik Estimasi Model Hybrid ARIMA-SVR Koin {selected_crypto}", fontsize=14)
                ax.set_xlabel("Tanggal", fontsize=11)
                ax.set_ylabel("Harga (USD)", fontsize=11)
                ax.legend()
                ax.grid(True, linestyle=':', alpha=0.6)
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Gagal memproses model Hybrid. Detail Masalah: {e}")
else:
    st.warning(f"Data untuk simbol '{selected_crypto}' gagal dimuat.")
