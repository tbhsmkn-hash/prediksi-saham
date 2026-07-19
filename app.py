import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, root_mean_squared_error

# 1. Daftar Koin (Gunakan -USD untuk konsistensi data di Yahoo Finance)
crypto_map = {
    "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD",
    "Pepe": "PEPE-USD", "Binance Coin": "BNB-USD", "Ripple": "XRP-USD",
    "Cardano": "ADA-USD", "Dogecoin": "DOGE-USD", "Shiba Inu": "SHIB-USD",
    "Avalanche": "AVAX-USD"
}

@st.cache_data(ttl=3600)
def get_data_with_conversion(ticker, duration, unit):
    try:
        # Pemetaan durasi ke format yfinance
        # Jika jam/hari, interval 1 jam/1 hari. Jika minggu/bulan/tahun, interval 1 hari.
        if unit == "Jam":
            period, interval = f"{duration}d", "1h"
        elif unit == "Hari":
            period, interval = f"{duration}d", "1d"
        elif unit == "Minggu":
            period, interval = f"{duration * 7}d", "1d"
        elif unit == "Bulan":
            period, interval = f"{duration * 30}d", "1d"
        else: # Tahun
            period, interval = f"{duration * 365}d", "1d"

        tk = yf.Ticker(ticker)
        df = tk.history(period=period, interval=interval)
        
        kurs_df = yf.Ticker("IDR=X").history(period="1d")
        kurs = float(kurs_df['Close'].iloc[-1])
        
        if df.empty: return None
        
        df['Close'] = df['Close'] * kurs
        return df[['Close']].dropna()
    except Exception as e:
        st.error(f"Error: {e}")
        return None
    try:
        # A. Menggunakan Ticker object (lebih stabil)
        tk = yf.Ticker(ticker)
        df = tk.history(period=f"{days}d")
        
        # B. Ambil kurs (menggunakan series .iloc[-1] agar jadi angka)
        kurs_df = yf.Ticker("IDR=X").history(period="1d")
        if kurs_df.empty:
            return None
        kurs = float(kurs_df['Close'].iloc[-1])
        
        if df.empty:
            st.warning(f"DEBUG: Data {ticker} kosong dari Yahoo.")
            return None
            
        # Konversi ke IDR
        df['Close'] = df['Close'] * kurs
        return df[['Close']].dropna()
        
    except Exception as e:
        st.error(f"DEBUG Error: {e}")
        return None
    try:
        # A. Ambil data koin (USD)
        df = yf.download(ticker, period=f"{days}d", interval="1d", progress=False, auto_adjust=True)
        
        # B. Ambil kurs IDR/USD untuk konversi
        kurs = yf.download("IDR=X", period="1d", progress=False)['Close'].iloc[-1]
        
        if df.empty:
            return None
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # C. Konversi harga ke IDR
        df['Close'] = df['Close'] * kurs
        return df[['Close']].dropna()
    except Exception as e:
        st.error(f"Gagal mengambil data: {e}")
        return None

# --- UI Sidebar ---
koin_nama = st.sidebar.selectbox("Pilih Koin:", list(crypto_map.keys()))
ticker = crypto_map[koin_nama]
st.sidebar.subheader("Pengaturan Waktu")
col1, col2 = st.sidebar.columns([1, 1])
with col1:
    durasi = st.number_input("Jumlah", min_value=1, value=30)
with col2:
    satuan = st.selectbox("Satuan", ["Hari", "Jam", "Minggu", "Bulan", "Tahun"])

# Panggil fungsi dengan parameter baru
df = get_data_with_conversion(ticker, durasi, satuan)

# --- Proses Utama ---
st.title(f"Analisis & Prediksi {koin_nama} (IDR)")
df = get_data_with_conversion(ticker, durasi, satuan)

if df is not None and not df.empty:
    st.line_chart(df)
    st.success(f"Berhasil menarik {len(df)} data historis untuk {koin_nama}.")
else:
    st.error("Data tetap tidak ditemukan. Coba ganti koin.")

# --- Lanjutan dari bagian UI (setelah st.line_chart(df)) ---

if st.button("🚀 Jalankan Analisis & Prediksi Model"):
    try:
        # 1. Persiapan Data (Pastikan menjadi array 1D)
        data = df['Close'].values.flatten()
        
        # 2. Pemodelan ARIMA
        # Menggunakan order default (1,1,1), Anda bisa menambah input parameter nanti
        model_arima = ARIMA(data, order=(1, 1, 1))
        model_arima_fitted = model_arima.fit()
        pred_arima = model_arima_fitted.fittedvalues
        
        # Hitung Residu (Error dari ARIMA)
        resid = data - pred_arima
        
        # 3. Pemodelan SVR pada Residu
        # Kita gunakan index waktu sebagai fitur input SVR
        X = np.arange(len(resid)).reshape(-1, 1)
        
        # Scaling agar SVR bekerja optimal
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(X)
        
        model_svr = SVR(kernel='rbf', C=100, gamma='scale')
        model_svr.fit(X_scaled, resid)
        pred_svr = model_svr.predict(X_scaled)
        
        # 4. Hasil Prediksi Hybrid
        hybrid_pred = pred_arima + pred_svr
        
        # 5. Menampilkan Metrik Evaluasi
        st.subheader("📊 Metrik Akurasi Model")
        col1, col2, col3 = st.columns(3)
        
        rmse = root_mean_squared_error(data, hybrid_pred)
        mae = mean_absolute_error(data, hybrid_pred)
        mape = mean_absolute_percentage_error(data, hybrid_pred) * 100
        
        col1.metric("RMSE", f"{rmse:,.0f}")
        col2.metric("MAE", f"{mae:,.0f}")
        col3.metric("MAPE", f"{mape:.2f}%")
        
        # 6. Visualisasi Fitting
        st.subheader("📉 Grafik Perbandingan")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, data, label="Harga Aktual", color="blue", alpha=0.6)
        ax.plot(df.index, hybrid_pred, label="Prediksi Hybrid (ARIMA-SVR)", color="red", linestyle="--")
        ax.legend()
        ax.set_title(f"Hasil Fitting Model Hybrid {koin_nama}")
        st.pyplot(fig)
        
        st.success("Prediksi berhasil dijalankan!")
        
    except Exception as err:
        st.error(f"Gagal melakukan kalkulasi model: {err}")