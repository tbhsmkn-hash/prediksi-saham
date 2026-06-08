import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt

# Konfigurasi Tampilan Web App
st.set_page_config(page_title="Sistem Hybrid ARIMA-SVR Indodax v2", layout="wide")

# ==========================================
# 1. PIPELINE DATA CORE API V2 RESMI INDODAX (BEBAS BLOKIR)
# ==========================================

def get_data_from_indodax(coin_symbol, days_back=365):
    """Menarik data historis harian murni menggunakan API Core V2 Resmi Indodax"""
    try:
        end_time = int(datetime.datetime.now().timestamp())
        start_time = end_time - (days_back * 24 * 60 * 60)
        
        # API V2 menggunakan huruf kecil dan underscore (Contoh: btc_idr)
        pair_ticker = f"{coin_symbol.lower()}_idr"
        
        # Endpoint Publik V2 Resmi untuk Candlestick harian (1 Hari = 1440 menit)
        url = f"https://indodax.com/api/v2/candles/{pair_ticker}?res=1440&from={start_time}&to={end_time}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            st.error(f"Gagal terhubung ke server Indodax. HTTP Status: {response.status_code}")
            return pd.DataFrame()
            
        json_data = response.json()
        
        # API V2 mengembalikan array di dalam array secara langsung: [[timestamp, open, high, low, close, volume], ...]
        if isinstance(json_data, list) and len(json_data) > 0:
            df = pd.DataFrame(json_data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            
            # Konversi timestamp detik ke format datetime
            df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')
            df.set_index('Date', inplace=True)
            df.drop('Timestamp', axis=1, inplace=True)
            df.sort_index(inplace=True)
            
            # Memastikan seluruh data numerik berupa float
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        else:
            st.error(f"API Indodax merespons kosong. Format data tidak sesuai untuk pair {pair_ticker}.")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Gagal memproses struktur data API Indodax V2: {e}")
        return pd.DataFrame()

# ==========================================
# 2. PANEL SIDEBAR (KONFIGURASI PARAMETER)
# ==========================================

st.sidebar.title("⚙️ Pengaturan Parameter")

st.sidebar.subheader("1. Sumber Data Pasar")
list_koin = ["BTC", "ETH", "BNB", "SOL", "XRP","ADA", "DOGE", "SHIB", "AVAX", "DOT","MATIC", "LINK", "TRX", "UNI", "ICP","BCH", "NEAR", "FIL", "LTC", "LEO","ETC", "APT",
			 "ATOM", "XLM", "HBAR","IMX", "GRT", "STX", "KAS", "TON","PEPE", "WIF", "FLOKI", "RUNE", "RENDER","THETA", "FTM", "LUNC", "AAVE", "ALGO","EGLD", "FLOW",
			 "MKR", "SAND", "MANA","XTZ", "VET", "AXS", "EOS", "NEO"]
koin_terpilih = st.sidebar.selectbox("Pilih Koin Kripto (Pair IDR Resmi Indodax):", list_koin)

rentang_hari = st.sidebar.slider("Rentang Data Historis (Hari):", 180, 730, 365)

st.sidebar.subheader("2. Orde Model Linier (ARIMA)")
p = st.sidebar.number_input("Orde AR (p):", min_value=0, max_value=5, value=1)
d = st.sidebar.number_input("Differencing (d):", min_value=0, max_value=2, value=1)
q = st.sidebar.number_input("Orde MA (q):", min_value=0, max_value=5, value=1)

st.sidebar.subheader("3. Konfigurasi SVR Residu")
kernel = st.sidebar.selectbox("Fungsi Kernel SVR:", ["rbf", "linear", "poly"])
c_param = st.sidebar.number_input("Nilai Regularisasi C:", min_value=1.0, max_value=2000.0, value=100.0, step=50.0)
gamma = st.sidebar.selectbox("Nilai Gamma SVR:", ["scale", "auto"])

st.sidebar.subheader("4. Pengaturan Proyeksi")
hari_prediksi = st.sidebar.slider("Durasi Peramalan Ke Depan (Hari):", 1, 30, 7)

# ==========================================
# 3. AREA DASHBOARD HALAMAN UTAMA
# ==========================================

st.title("📊 Platform Komputasi Cerdas Hybrid ARIMA-SVR (Indodax API V2)")
st.markdown(f"Analisis integrasi regresi klasik linier **ARIMA** dengan optimasi pola non-linier residu menggunakan **SVR** menggunakan data riil **PT. Indodax Nasional Indonesia**.")

# Eksekusi penarikan data secara langsung dari API Core V2 Indodax
with st.spinner("Mengunduh data transaksi harian langsung dari core server Indodax..."):
    df_crypto = get_data_from_indodax(koin_terpilih, days_back=rentang_hari)
    mata_uang = "IDR"

if not df_crypto.empty:
    
    # --- BAGIAN A: EKSPLORASI DATA TERBARU & TOMBOL DOWNLOAD ---
    col_tabel, col_unduh = st.columns([3, 1])
    with col_tabel:
        st.subheader(f"📋 10 Baris Data Transaksi Terakhir Rupiah ({koin_terpilih}/IDR)")
        st.dataframe(df_crypto.tail(10))
    with col_unduh:
        st.markdown("<br><br>", unsafe_allow_html=True)
        csv_buffer = df_crypto.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Unduh File Dataset (.CSV)",
            data=csv_buffer,
            file_name=f"dataset_indodax_{koin_terpilih}_{datetime.date.today()}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # --- BAGIAN B: ANALISIS STATISTIK DESKRIPTIF ---
    st.subheader("📊 Analisis Statistik Deskriptif Nilai Rupiah")
    col_stat_df, col_stat_box = st.columns([2, 1])
    with col_stat_df:
        st.dataframe(df_crypto.describe().T)
    with col_stat_box:
        st.markdown(
            f"<div style='padding:15px; border-radius:10px; background-color:#1e293b; color:white'>"
            f"<b>Metadata Observasi Indodax:</b><br>"
            f"• Titik Awal Tren: {df_crypto.index.min().strftime('%d-%m-%Y')}<br>"
            f"• Titik Akhir Tren: {df_crypto.index.max().strftime('%d-%m-%Y')}<br>"
            f"• Jumlah Sampel: {len(df_crypto)} Hari Efektif"
            f"</div>", 
            unsafe_allow_html=True
        )

    # --- BAGIAN C: PENGKONDISIAN TOMBOL EKSEKUSI PREDIKSI ---
    st.markdown("---")
    st.subheader("🔮 Algoritma Komputasi Prediksi")
    st.info("Tekan tombol di bawah ini untuk mengaktifkan mesin latih model ARIMA, SVR, dan Kombinasi Hybrid secara simultan.")
    
    if st.button("🚀 Jalankan Analisis & Prediksi Model", type="primary", use_container_width=True):
        try:
            data_close = df_crypto['Close'].astype(float)
            
            # 1. PEMODELAN LINIER: ARIMA MURNI
            model_arima = ARIMA(data_close, order=(p, d, q))
            model_arima_fitted = model_arima.fit()
            prediksi_arima_murni = model_arima_fitted.fittedvalues
            residu_galat_arima = data_close - prediksi_arima_murni
            
            # 2. PEMODELAN NON-LINIER: TRAINING SVR BERDASARKAN LAG RESIDU
            df_residu_lag = pd.DataFrame({'Residu': residu_galat_arima})
            df_residu_lag['Lag_1'] = df_residu_lag['Residu'].shift(1)
            df_residu_lag.dropna(inplace=True)
            
            X_svr = df_residu_lag[['Lag_1']].values
            y_svr = df_residu_lag['Residu'].values
            
            scaler_input = MinMaxScaler(feature_range=(-1, 1))
            X_svr_scaled = scaler_input.fit_transform(X_svr)
            
            model_svr = SVR(kernel=kernel, C=c_param, gamma=gamma)
            model_svr.fit(X_svr_scaled, y_svr)
            prediksi_residu_svr = model_svr.predict(X_svr_scaled)
            
            # 3. SINKRONISASI INDEKS BARIS
            data_aktual_sinkron = data_close.iloc[1:]
            prediksi_arima_sinkron = prediksi_arima_murni.iloc[1:]
            
            hasil_prediksi_hybrid = prediksi_arima_sinkron + prediksi_residu_svr
            hasil_prediksi_svr_only = data_close.shift(1).iloc[1:] + prediksi_residu_svr

            # --- BAGIAN D: TABEL PERBANDINGAN TINGKAT EROR ---
            st.subheader("📉 Tabel Perbandingan Akurasi Validasi Model")
            
            df_matriks_eror = pd.DataFrame({
                "Arsitektur Model": ["ARIMA Model (Linier)", "SVR Model (Non-Linier Residu)", "Model Gabungan Hybrid (ARIMA-SVR)"],
                "RMSE": [
                    root_mean_squared_error(data_aktual_sinkron, prediksi_arima_sinkron),
                    root_mean_squared_error(data_aktual_sinkron, hasil_prediksi_svr_only),
                    root_mean_squared_error(data_aktual_sinkron, hasil_prediksi_hybrid)
                ],
                "MAE": [
                    mean_absolute_error(data_aktual_sinkron, prediksi_arima_sinkron),
                    mean_absolute_error(data_aktual_sinkron, hasil_prediksi_svr_only),
                    mean_absolute_error(data_aktual_sinkron, hasil_prediksi_hybrid)
                ],
                "MAPE (%)": [
                    mean_absolute_percentage_error(data_aktual_sinkron, prediksi_arima_sinkron) * 100,
                    mean_absolute_percentage_error(data_aktual_sinkron, hasil_prediksi_svr_only) * 100,
                    mean_absolute_percentage_error(data_aktual_sinkron, hasil_prediksi_hybrid) * 100
                ]
            })
            st.table(df_matriks_eror)
            
            # --- BAGIAN E: VISUALISASI GRAFIK FITTING ---
            st.subheader("📊 Visualisasi Hasil Fitting Nilai Historis")
            
            fig_hist, ax_hist = plt.subplots(figsize=(15, 6))
            ax_hist.plot(data_close.index, data_close.values, label="Harga Pasar Aktual (Indodax)", color="darkcyan", linewidth=2.5)
            ax_hist.plot(prediksi_arima_sinkron.index, prediksi_arima_sinkron.values, label="Estimasi ARIMA", color="orange", linestyle="--", alpha=0.8)
            ax_hist.plot(data_aktual_sinkron.index, hasil_prediksi_hybrid, label="Estimasi Hybrid (ARIMA-SVR)", color="crimson", linewidth=2)
            
            ax_hist.set_title(f"Grafik Komparasi Pergerakan Harga vs Estimasi Model Finansial ({koin_terpilih}/IDR)", fontsize=13)
            ax_hist.set_xlabel("Skala Waktu (Tanggal)", fontsize=11)
            ax_hist.set_ylabel(f"Nilai Penutupan Aset ({mata_uang})", fontsize=11)
            ax_hist.grid(True, linestyle=":", alpha=0.5)
            ax_hist.legend(loc="upper left")
            
            ax_hist.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
            st.pyplot(fig_hist)
            
            # --- BAGIAN F: PROYEKSI PERAMALAN MASA DEPAN ---
            st.markdown("---")
            st.subheader(f"🔮 Hasil Proyeksi Peramalan Tren Masa Depan ({hari_prediksi} Hari Kedepan)")
            
            proyeksi_raw_arima = model_arima_fitted.forecast(steps=hari_prediksi)
            
            residu_terakhir = y_svr[-1]
            list_proyeksi_residu_svr = []
            
            for _ in range(hari_prediksi):
                scaled_lag_input = scaler_input.transform([[residu_terakhir]])
                output_pred_residu = model_svr.predict(scaled_lag_input)[0]
                list_proyeksi_residu_svr.append(output_pred_residu)
                residu_terakhir = output_pred_residu
                
            nilai_akhir_proyeksi_hybrid = proyeksi_raw_arima.values + np.array(list_proyeksi_residu_svr)
            
            tanggal_akhir_pasar = data_close.index[-1]
            indeks_tanggal_proyeksi = pd.date_range(start=tanggal_akhir_pasar + datetime.timedelta(days=1), periods=hari_prediksi)
            
            df_output_ramalan = pd.DataFrame({
                "Proyeksi ARIMA (Linier)": proyeksi_raw_arima.values,
                "Koreksi Residu SVR": list_proyeksi_residu_svr,
                f"Hasil Akhir Hybrid ({mata_uang})": nilai_akhir_proyeksi_hybrid
            }, index=indeks_tanggal_proyeksi)
            df_output_ramalan.index.name = "Tanggal Proyeksi"
            
            col_res_tabel, col_res_grafik = st.columns([1, 1])
            with col_res_tabel:
                st.markdown("**Tabel Rincian Angka Nilai Proyeksi:**")
                st.dataframe(df_output_ramalan)
            with col_res_grafik:
                st.markdown("**Grafik Tren Proyeksi Masa Depan:**")
                fig_f, ax_f = plt.subplots(figsize=(10, 5))
                ax_f.plot(df_output_ramalan.index, df_output_ramalan[f"Hasil Akhir Hybrid ({mata_uang})"], marker='o', linewidth=2, color='crimson', label='Proyeksi Hybrid')
                ax_f.plot(df_output_ramalan.index, df_output_ramalan["Proyeksi ARIMA (Linier)"], marker='x', linestyle='--', color='orange', label='Proyeksi ARIMA Murni')
                ax_f.set_ylabel(f"Harga ({mata_uang})")
                ax_f.grid(True, linestyle=":", alpha=0.5)
                ax_f.legend()
                plt.xticks(rotation=45)
                st.pyplot(fig_f)
                
        except Exception as err:
            st.error(f"Gagal melakukan kalkulasi matriks model: {err}")
            
else:
    st.warning("Saluran data kosong. Gagal memuat data dari API Core V2 Indodax.")
