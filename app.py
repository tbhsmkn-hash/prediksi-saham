import streamlit as st
import pandas as pd
import numpy as np
import time
from statsmodels.tsa.arima.model import ARIMA
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import datetime
import requests

# --- Pengaturan Halaman Utama ---
st.title("Aplikasi Prediksi Cryptocurrency Berbasis Hybrid ARIMA-SVR")
st.write("Aplikasi ini mengombinasikan data otomatis dari Google Sheets, model klasik ARIMA (Linear), dan Machine Learning SVR (Non-Linear).")

# --- DAFTAR 50 COIN CRYPTO TERPOPULER ---
# --- DAFTAR COIN CRYPTO YANG STABIL DI GOOGLE FINANCE ---
CRYPTO_LIST = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", 
    "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "LINK-USD", 
    "TRX-USD", "UNI-USD", "ICP-USD", "BCH-USD", "NEAR-USD", 
    "FIL-USD", "LTC-USD", "ETC-USD", "APT-USD", "ATOM-USD", 
    "XLM-USD", "HBAR-USD", "GRT-USD", "STX-USD", "TON-USD",
    "THETA-USD", "FTM-USD", "AAVE-USD", "ALGO-USD", "EGLD-USD", 
    "FLOW-USD", "MKR-USD", "SAND-USD", "MANA-USD", "XTZ-USD", 
    "VET-USD", "AXS-USD", "EOS-USD", "NEO-USD", "LUNC-USD"
]

# --- Pengaturan Data di Sidebar ---
st.sidebar.header("1. Pengaturan Data Crypto")
selected_crypto = st.sidebar.selectbox("Pilih Koin Crypto", CRYPTO_LIST)
selected_start_date = st.sidebar.date_input("Tanggal Mulai (Baseline)", datetime.date(2021, 1, 1))
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

# ==========================================================
# PEMBAHASAN TERAKHIR: PEMUTAKHIRAN PIPELINE GOOGLE SHEETS
# ==========================================================

def trigger_google_sheets_sync(ticker_name):
    """Menyuruh Google Sheets memperbarui data historis via Apps Script Web App."""
    WEB_APP_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRLbUkprSxIP60BPBsTGG12NyDaqUEhGWoXER4jKMdETNTAjGUaCv913PQVEgABXf_37hdmU4spLl34/pub?gid=0&single=true&output=csv"
    try:
        # Mengirimkan parameter nama koin secara dinamis (?ticker=BTC-USD)
        requests.get(f"{WEB_APP_URL}?ticker={ticker_name}")
    except Exception as e:
        st.warning(f"Gagal memicu sinkronisasi cloud: {e}")

@st.cache_data(ttl=5) # Setel cache sangat rendah (5 detik) agar data koin baru tidak tersangkut cache lama
def get_crypto_data_from_sheets(ticker_name):
    """Membaca data historis dari Google Sheets dengan sistem jeda sinkronisasi cloud."""
    
    # 1. Kirim sinyal ke Google Apps Script Web App untuk mengganti data koin di Sheets
    trigger_google_sheets_sync(ticker_name)
    
    # 2. BERI JEDA TUNGGU (PENTING): Memberikan waktu 4 detik agar Google Sheets selesai memuat formula
    time.sleep(4)
    
    # 3. URL Publish to Web CSV Anda yang sudah diperbaiki
    SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRLbUkprSxIP60BPBsTGG12NyDaqUEhGWoXER4jKMdETNTAjGUaCv913PQVEgABXf_37hdmU4spLl34/pub?output=csv"
    
    try:
        df = pd.read_csv(SHEET_CSV_URL)
        
        if df.empty or len(df) < 5:
            st.error(f"Google Sheets mendeteksi data '{ticker_name}' masih kosong atau dalam proses memuat di server Google. Coba lakukan refresh beberapa saat lagi.")
            return pd.DataFrame()

        # --- NORMALISASI KOLOM MULTI-BAHASA (INGGRIS / INDONESIA) ---
        df.columns = [str(col).strip().lower() for col in df.columns]
        
        # Deteksi Kolom Tanggal
        if 'date' in df.columns:
            df.rename(columns={'date': 'Date'}, inplace=True)
        elif 'tanggal' in df.columns:
            df.rename(columns={'tanggal': 'Date'}, inplace=True)
        else:
            df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
            
        # Deteksi Kolom Close / Tutup
        if 'close' in df.columns:
            df.rename(columns={'close': 'Close'}, inplace=True)
        elif 'tutup' in df.columns:
            df.rename(columns={'tutup': 'Close'}, inplace=True)
            
        # Deteksi Kolom Lainnya
        kolom_kamus = {
            'open': 'Open', 'buka': 'Open',
            'high': 'High', 'tinggi': 'High',
            'low': 'Low', 'rendah': 'Low',
            'volume': 'Volume'
        }
        df.rename(columns=kolom_kamus, inplace=True)

        # Konversi Tanggal
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        # Konversi ke Numerik
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Gagal memuat data dari Google Sheets. Detail Masalah: {e}")
        return pd.DataFrame()
# --- Eksekusi Pengambilan Data terintegrasi Google Sheets ---
df_crypto = get_crypto_data_from_sheets(selected_crypto)

# --- Alur Tampilan Utama Aplikasi ---
if not df_crypto.empty and 'Close' in df_crypto.columns:
    st.subheader(f"Data Historis Harga {selected_crypto} (Sumber: Pipeline Google Sheets)")
    st.write(df_crypto[['Open', 'High', 'Low', 'Close', 'Volume']].tail()) 
    st.line_chart(df_crypto['Close']) 
    
    # --- Tombol Eksekusi Model Hybrid ---
    if st.button("Jalankan Prediksi Hybrid ARIMA-SVR"):
        with st.spinner(f"Melatih model Hybrid untuk {selected_crypto}..."):
            try:
                # Sinkronisasi interval harian (DAILY)
                series_close = df_crypto['Close'].asfreq('D')
                if series_close.isnull().any():
                    series_close = series_close.ffill()
                
                # ==========================================
                # LENGKAH 1: TRAINING MODEL ARIMA (LINEAR)
                # ==========================================
                model_arima = ARIMA(series_close, order=(p, d, q))
                model_arima_fit = model_arima.fit()
                arima_fitted = model_arima_fit.fittedvalues
                residuals = series_close - arima_fitted
                
                # ==========================================
                # LANGKAH 2: TRAINING MODEL SVR (NON-LINEAR)
                # ==========================================
                look_back = 3
                X_svr, y_svr = [], []
                res_values = residuals.values
                
                for i in range(len(res_values) - look_back):
                    X_svr.append(res_values[i:(i + look_back)])
                    y_svr.append(res_values[i + look_back])
                
                X_svr, y_svr = np.array(X_svr), np.array(y_svr)
                
                scaler = StandardScaler()
                X_svr_scaled = scaler.fit_transform(X_svr)
                
                model_svr = SVR(kernel=svr_kernel, C=svr_c)
                model_svr.fit(X_svr_scaled, y_svr)
                
                # Mendapatkan fitted values dari SVR untuk data historis
                svr_fitted_res = model_svr.predict(X_svr_scaled)
                
                # Keselarasan panjang data aktual untuk evaluasi
                aktual_evaluasi = series_close.values[look_back:]
                arima_evaluasi = arima_fitted.values[look_back:]
                hybrid_evaluasi = arima_evaluasi + svr_fitted_res
                
                # ==========================================
                # HITUNG METRIK EVALUASI (RMSE, MAE, MAPE)
                # ==========================================
                rmse_arima = np.sqrt(mean_squared_error(aktual_evaluasi, arima_evaluasi))
                mae_arima = mean_absolute_error(aktual_evaluasi, arima_evaluasi)
                mape_arima = mean_absolute_percentage_error(aktual_evaluasi, arima_evaluasi) * 100
                
                residual_aktual = res_values[look_back:]
                rmse_svr = np.sqrt(mean_squared_error(residual_aktual, svr_fitted_res))
                mae_svr = mean_absolute_error(residual_aktual, svr_fitted_res)
                mape_svr = mean_absolute_percentage_error(residual_aktual, svr_fitted_res) * 100
                
                rmse_hybrid = np.sqrt(mean_squared_error(aktual_evaluasi, hybrid_evaluasi))
                mae_hybrid = mean_absolute_error(aktual_evaluasi, hybrid_evaluasi)
                mape_hybrid = mean_absolute_percentage_error(aktual_evaluasi, hybrid_evaluasi) * 100

                # Tampilkan Skor Performa di Halaman Utama Web
                st.subheader("📊 Evaluasi Performa Model (Data Historis)")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    **Model ARIMA**
                    * RMSE: `{rmse_arima:.4f}`
                    * MAE: `{mae_arima:.4f}`
                    * MAPE: `{mape_arima:.2f}%`
                    """)
                with col2:
                    st.markdown(f"""
                    **Model SVR (Residual)**
                    * RMSE: `{rmse_svr:.4f}`
                    * MAE: `{mae_svr:.4f}`
                    * MAPE: `{mape_svr:.2f}%`
                    """)
                with col3:
                    st.markdown(f"""
                    **Model Hybrid (ARIMA-SVR)**
                    * RMSE: `{rmse_hybrid:.4f}`
                    * MAE: `{mae_hybrid:.4f}`
                    * **MAPE: `{mape_hybrid:.2f}%`**
                    """)
                
                # ==========================================
                # LANGKAH 3: PROSES PERAMALAN (FORECASTING MASA DEPAN)
                # ==========================================
                arima_forecast = model_arima_fit.forecast(steps=forecast_steps)
                
                svr_forecast = []
                input_res = list(res_values[-look_back:])
                
                for _ in range(forecast_steps):
                    input_res_scaled = scaler.transform([input_res[-look_back:]])
                    pred_res = model_svr.predict(input_res_scaled)[0]
                    svr_forecast.append(pred_res)
                    input_res.append(pred_res)
                
                hybrid_forecast = arima_forecast.values + np.array(svr_forecast)
                
                # ==========================================
                # LANGKAH 4: VISUALISASI DAN OUTPUT CSV
                # ==========================================
                last_date = series_close.index[-1]
                forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
                
                df_forecast = pd.DataFrame({
                    'Prediksi ARIMA (Linear)': arima_forecast.values,
                    'Koreksi SVR (Non-Linear)': svr_forecast,
                    'Total Prediksi Hybrid ($)': hybrid_forecast
                }, index=forecast_index)
                df_forecast.index.name = 'Tanggal'
                
                st.subheader(f"🔮 Hasil Peramalan {forecast_steps} Hari ke Depan")
                st.write(df_forecast)
                
                # Integrasi Tombol Download CSV
                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=True).encode('utf-8')
                
                csv_data = convert_df_to_csv(df_forecast)
                
                st.download_button(
                    label="📥 Unduh Hasil Prediksi (.CSV)",
                    data=csv_data,
                    file_name=f"prediksi_hybrid_{selected_crypto}_{last_date.strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="download-csv"
                )
                
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
    st.warning(f"Data untuk simbol '{selected_crypto}' gagal dimuat. Pastikan ID CSV Publish to Web Google Sheets sudah benar.")
