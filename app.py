import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt

# Konfigurasi Halaman Web App
st.set_page_config(page_title="Analisis Hybrid ARIMA-SVR Crypto", layout="wide")

# ==========================================
# 1. DATA PIPELINE (PERBAIKAN CORE API INDODAX)
# ==========================================

def get_data_from_indodax(coin_symbol, days_back=365):
    """Menarik data historis harian dari API Publik Core v2 Indodax (Bebas Blokir)"""
    try:
        # Konversi waktu ke Unix Timestamp (Detik)
        end_time = int(datetime.datetime.now().timestamp())
        start_time = end_time - (days_back * 24 * 60 * 60)
        
        # Format Pair Koin di API v2 Indodax menggunakan huruf kecil (contoh: btc_idr)
        pair = f"{coin_symbol.lower()}_idr"
        
        # Menggunakan Endpoint API v2 resmi khusus Candlestick (Resolusi 1D = 1 Hari = 1440 menit)
        url = f"https://indodax.com/api/v2/candles/{pair}?res=1440&from={start_time}&to={end_time}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            st.error(f"Koneksi Indodax Gagal. Status Code: {response.status_code}")
            return pd.DataFrame()
            
        data_json = response.json()
        
        # API v2 Indodax mengembalikan array of arrays: [timestamp, open, high, low, close, volume]
        if isinstance(data_json, list) and len(data_json) > 0:
            df = pd.DataFrame(data_json, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            
            # Konversi unix timestamp (milidetik jika panjangnya 13 digit, detik jika 10 digit)
            df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')
            df.set_index('Date', inplace=True)
            
            # Drop kolom timestamp mentah dan urutkan tren waktu
            df.drop('Timestamp', axis=1, inplace=True)
            df.sort_index(inplace=True)
            
            # Memastikan seluruh data berupa tipe data numerik mengambang (float)
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            return df
        else:
            st.error("Format respons data dari Indodax tidak sesuai atau kosong.")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Gagal memproses parsing API Indodax: {e}")
        return pd.DataFrame()

def get_data_from_yahoo(ticker_name, days_back=365):
    """Menarik data historis harian dari Yahoo Finance API (USD)"""
    try:
        hari_ini = datetime.date.today()
        tanggal_mulai = hari_ini - datetime.timedelta(days=days_back)
        
        df = yf.download(ticker_name, start=tanggal_mulai.strftime("%Y-%m-%d"), end=hari_ini.strftime("%Y-%m-%d"), progress=False)
        
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
                
            df.index = pd.to_datetime(df.index)
            df.index.name = 'Date'
            df.sort_index(inplace=True)
            
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Gagal memuat data dari Yahoo Finance: {e}")
        return pd.DataFrame()

# ==========================================
# 2. PANEL SIDERBAR (KONFIGURASI PARAMETER)
# ==========================================

st.sidebar.title("⚙️ Panel Konfigurasi")

st.sidebar.subheader("1. Pengaturan Sumber Data")
sumber_data = st.sidebar.selectbox(
    "Pilih Bursa Acuan:",
    ["Indodax Bursa Indonesia (IDR)", "Yahoo Finance (USD)"]
)

if sumber_data == "Yahoo Finance (USD)":
    list_koin = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", 
		"ADA-USD", "DOGE-USD", "SHIB-USD", "AVAX-USD", "DOT-USD",
		"MATIC-USD", "LINK-USD", "TRX-USD", "UNI-USD", "ICP-USD",
		"BCH-USD", "NEAR-USD", "FIL-USD", "LTC-USD", "LEO-USD",
		"ETC-USD", "APT-USD", "ATOM-USD", "XLM-USD", "HBAR-USD",
		"IMX-USD", "GRT-USD", "STX-USD", "KAS-USD", "TON-USD",
		"PEPE-USD", "WIF-USD", "FLOKI-USD", "RUNE-USD", "RENDER-USD",
		"THETA-USD", "FTM-USD", "LUNC-USD", "AAVE-USD", "ALGO-USD",
		"EGLD-USD", "FLOW-USD", "MKR-USD", "SAND-USD", "MANA-USD",
		"XTZ-USD", "VET-USD", "AXS-USD", "EOS-USD", "NEO-USD"]
    koin_terpilih = st.sidebar.selectbox("Pilih Cryptocurrency:", list_koin)
else:
    list_koin = ["BTC", "ETH", "BNB", "SOL", "XRP", 
		"ADA", "DOGE", "SHIB", "AVAX", "DOT",
		"MATIC", "LINK", "TRX", "UNI", "ICP",
		"BCH", "NEAR", "FIL", "LTC", "LEO",
		"ETC", "APT", "ATOM", "XLM", "HBAR",
		"IMX", "GRT", "STX", "KAS", "TON",
		"PEPE", "WIF", "FLOKI", "RUNE", "RENDER",
		"THETA", "FTM", "LUNC", "AAVE", "ALGO",
		"EGLD", "FLOW", "MKR", "SAND", "MANA",
		"XTZ", "VET", "AXS", "EOS", "NEO"]
    koin_terpilih = st.sidebar.selectbox("Pilih Cryptocurrency (Pair IDR):", list_koin)

rentang_hari = st.sidebar.slider("Rentang Pengambilan Data (Hari):", 180, 730, 365)

st.sidebar.subheader("2. Komponen Linier (ARIMA)")
p = st.sidebar.number_input("Orde Autoregressive (p):", min_value=0, max_value=5, value=1)
d = st.sidebar.number_input("Orde Differencing (d):", min_value=0, max_value=2, value=1)
q = st.sidebar.number_input("Orde Moving Average (q):", min_value=0, max_value=5, value=1)

st.sidebar.subheader("3. Komponen Non-Linier Residu (SVR)")
kernel = st.sidebar.selectbox("Fungsi Kernel SVR:", ["rbf", "linear", "poly"])
c_param = st.sidebar.number_input("Parameter Regularisasi C:", min_value=1.0, max_value=2000.0, value=100.0, step=50.0)
gamma = st.sidebar.selectbox("Parameter Gamma SVR:", ["scale", "auto"])

st.sidebar.subheader("4. Pengaturan Proyeksi")
hari_prediksi = st.sidebar.slider("Durasi Peramalan Ke Depan (Hari):", 1, 30, 7)

# ==========================================
# 3. HALAMAN DASHBOARD UTAMA
# ==========================================

st.title("📈 Sistem Komputasi Cerdas Hybrid ARIMA-SVR")
st.markdown("Aplikasi Analitik Peramalan Harga Cryptocurrency Menggunakan SVR Untuk Pemodelan Galat/Residu Hasil Proyeksi ARIMA.")

# Memulai Alur Penarikan Data Otomatis Berdasarkan Input Bursa
with st.spinner("Menghubungkan ke API Bursa untuk menyinkronkan data..."):
    if sumber_data == "Yahoo Finance (USD)":
        df_crypto = get_data_from_yahoo(koin_terpilih, days_back=rentang_hari)
        mata_uang = "USD"
    else:
        df_crypto = get_data_from_indodax(koin_terpilih, days_back=rentang_hari)
        mata_uang = "IDR"

# Cek Apakah Pipeline Berhasil Mengisi Dataframe
if not df_crypto.empty:
    
    # --- BAGIAN A: EKSPLORASI DATASET HISTORIS ---
    col_tabel, col_unduh = st.columns([3, 1])
    with col_tabel:
        st.subheader(f"📋 10 Data Historis Transaksi Terbaru ({koin_terpilih})")
        st.dataframe(df_crypto.tail(10))
    with col_unduh:
        st.markdown("<br><br>", unsafe_transform=True)
        # Tombol Unduh Eksport Data CSV
        csv_buffer = df_crypto.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Unduh File Dataset (.CSV)",
            data=csv_buffer,
            file_name=f"dataset_{koin_terpilih}_{datetime.date.today()}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # --- BAGIAN B: ANALISIS STATISTIK (Statistical Analysis) ---
    st.subheader("📊 Analisis Statistik Deskriptif Pasar")
    col_stat_tabel, col_stat_info = st.columns([2, 1])
    with col_stat_tabel:
        st.dataframe(df_crypto.describe().T)
    with col_stat_info:
        st.markdown(
            f"<div style='padding:15px; border-radius:10px; background-color:#f0f2f6; color:black'>"
            f"<b>Metadata Log Jaringan:</b><br>"
            f"• Rentang Awal: {df_crypto.index.min().strftime('%d %B %Y')}<br>"
            f"• Rentang Akhir: {df_crypto.index.max().strftime('%d %B %Y')}<br>"
            f"• Total Data Observasi: {len(df_crypto)} Hari Aktual"
            f"</div>", 
            unsafe_allow_html=True
        )

    # --- BAGIAN C: PENGKONDISIAN TOMBOL EKSEKUSI PREDIKSI ---
    st.markdown("---")
    st.subheader("⚡ Pemrosesan Algoritma Machine Learning")
    st.warning("Sesuaikan parameter matriks di sidebar kiri terlebih dahulu. Klik tombol di bawah ini untuk memulai pelatihan model.")
    
    # Pemicu Pemodelan Matematika
    if st.button("🚀 Jalankan Analisis & Prediksi Model", type="primary", use_container_width=True):
        try:
            # Isolasi Target Variabel (Harga Penutupan / Close)
            data_close = df_crypto['Close'].astype(float)
            
            # 1. TAHAP LINIER: EKSEKUSI MODEL ARIMA MURNI
            model_arima = ARIMA(data_close, order=(p, d, q))
            model_arima_fitted = model_arima.fit()
            prediksi_arima_murni = model_arima_fitted.fittedvalues
            
            # Hitung Nilai Galat / Residu Linier
            residu_galat_arima = data_close - prediksi_arima_murni
            
            # 2. TAHAP NON-LINIER: TRAINING SVR BERDASARKAN LAG RESIDU
            df_residu_lag = pd.DataFrame({'Residu': residu_galat_arima})
            df_residu_lag['Lag_1'] = df_residu_lag['Residu'].shift(1)
            df_residu_lag.dropna(inplace=True)
            
            X_svr = df_residu_lag[['Lag_1']].values
            y_svr = df_residu_lag['Residu'].values
            
            # Skalasi Fitur Residu agar optimal pada Struktur Fungsi Kernel SVR
            scaler_input = MinMaxScaler(feature_range=(-1, 1))
            X_svr_scaled = scaler_input.fit_transform(X_svr)
            
            model_svr = SVR(kernel=kernel, C=c_param, gamma=gamma)
            model_svr.fit(X_svr_scaled, y_svr)
            prediksi_residu_svr = model_svr.predict(X_svr_scaled)
            
            # 3. SINKRONISASI INDEKS BARIS EVALUASI (Bypass Baris Pertama karena Lag Efek)
            data_aktual_sinkron = data_close.iloc[1:]
            prediksi_arima_sinkron = prediksi_arima_murni.iloc[1:]
            
            # Nilai Prediksi Akhir Gabungan Hybrid
            hasil_prediksi_hybrid = prediksi_arima_sinkron + prediksi_residu_svr
            
            # Rekonstruksi Nilai SVR Murni ke Skala Harga Aktual (Sebagai Pembanding Independen)
            hasil_prediksi_svr_only = data_close.shift(1).iloc[1:] + prediksi_residu_svr

            # --- BAGIAN D: TABEL KOMPARASI EVALUASI ERROR ---
            st.subheader("📉 Tabel Perbandingan Tingkat Kesalahan (Error Evaluation)")
            
            df_evaluasi_error = pd.DataFrame({
                "Nama Model Analitis": ["ARIMA Murni (Linear Model)", "SVR Murni (Non-Linear Residu)", "Kombinasi Hybrid (ARIMA-SVR)"],
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
            st.table(df_evaluasi_error)
            
            # --- BAGIAN E: VISUALISASI HISTORIS (Visualization) ---
            st.subheader("📊 Visualisasi Hasil Fitting Model")
            
            fig_hist, ax_hist = plt.subplots(figsize=(15, 6))
            ax_hist.plot(data_close.index, data_close.values, label="Harga Aktual Pasar", color="teal", linewidth=2.5)
            ax_hist.plot(prediksi_arima_sinkron.index, prediksi_arima_sinkron.values, label="Estimasi ARIMA", color="orange", linestyle="--", alpha=0.8)
            ax_hist.plot(data_aktual_sinkron.index, hasil_prediksi_hybrid, label="Estimasi Hybrid (ARIMA-SVR)", color="purple", linewidth=2)
            
            ax_hist.set_title(f"Grafik Perbandingan Pergerakan Harga Aktual vs Hasil Estimasi Model ({koin_terpilih})", fontsize=13)
            ax_hist.set_xlabel("Garis Waktu (Tanggal)", fontsize=11)
            ax_hist.set_ylabel(f"Nilai Aset ({mata_uang})", fontsize=11)
            ax_hist.grid(True, linestyle=":", alpha=0.5)
            ax_hist.legend(loc="upper left")
            
            # Formatting bilangan ribuan untuk Rupiah
            if mata_uang == "IDR":
                ax_hist.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
                
            st.pyplot(fig_hist)
            
            # --- BAGIAN F: PROYEKSI NILAI MASA DEPAN ---
            st.markdown("---")
            st.subheader(f"🔮 Hasil Peramalan Tren Masa Depan ({hari_prediksi} Hari Ke Depan)")
            
            # Peramalan Masa Depan Menggunakan Koefisien ARIMA
            proyeksi_raw_arima = model_arima_fitted.forecast(steps=hari_prediksi)
            
            # Peramalan Autoregresif Masa Depan Menggunakan Bobot SVR
            residu_terakhir = y_svr[-1]
            list_proyeksi_residu_svr = []
            
            for _ in range(hari_prediksi):
                scaled_lag_input = scaler_input.transform([[residu_terakhir]])
                output_pred_residu = model_svr.predict(scaled_lag_input)[0]
                list_proyeksi_residu_svr.append(output_pred_residu)
                residu_terakhir = output_pred_residu # Update data lag input dengan prediksi terakhir
                
            # Akumulasikan Nilai Komponen Proyeksi
            nilai_akhir_proyeksi_hybrid = proyeksi_raw_arima.values + np.array(list_proyeksi_residu_svr)
            
            # Menentukan Penanggalan Indeks Masa Depan
            tanggal_paling_akhir = data_close.index[-1]
            indeks_tanggal_proyeksi = pd.date_range(start=tanggal_paling_akhir + datetime.timedelta(days=1), periods=hari_prediksi)
            
            # Konstruksi ke dalam Dataframe Penampil
            df_output_ramalan = pd.DataFrame({
                "Proyeksi ARIMA (Linier)": proyeksi_raw_arima.values,
                "Koreksi Residu SVR": list_proyeksi_residu_svr,
                f"Hasil Akhir Hybrid ({mata_uang})": nilai_akhir_proyeksi_hybrid
            }, index=indeks_tanggal_proyeksi)
            df_output_ramalan.index.name = "Tanggal Proyeksi"
            
            # Pemisahan Output Proyeksi ke Dalam Dua Kolom Visual
            col_res_tabel, col_res_grafik = st.columns([1, 1])
            with col_res_tabel:
                st.markdown("**Tabel Angka Rincian Proyeksi Masa Depan:**")
                st.dataframe(df_output_ramalan)
            with col_res_grafik:
                st.markdown("**Grafik Tren Proyeksi Masa Depan:**")
                fig_f, ax_f = plt.subplots(figsize=(10, 5))
                ax_f.plot(df_output_ramalan.index, df_output_ramalan[f"Hasil Akhir Hybrid ({mata_uang})"], marker='o', linewidth=2, color='purple', label='Proyeksi Hybrid')
                ax_f.plot(df_output_ramalan.index, df_output_ramalan["Proyeksi ARIMA (Linier)"], marker='x', linestyle='--', color='orange', label='Proyeksi ARIMA')
                ax_f.set_ylabel(f"Harga ({mata_uang})")
                ax_f.grid(True, linestyle=":", alpha=0.5)
                ax_f.legend()
                plt.xticks(rotation=45)
                st.pyplot(fig_f)
                
        except Exception as err:
            st.error(f"Gagal melakukan komputasi numerik model: {err}")
            st.info("Catatan: Terjadinya eror matematika matriks biasanya dipicu oleh ordo differencing (d) atau parameter ARIMA yang tidak stasioner dengan sebaran data. Sila ubah nilai parameter p, d, q pada sidebar.")
            
else:
    st.warning("Saluran data bursa kosong atau tidak merespons secara tepat. Silakan pilih opsi bursa penopang lainnya.")
