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

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Prediksi Crypto Hybrid ARIMA-SVR", layout="wide")

# ==========================================
# 1. FUNGSI PENARIKAN DATA (DENGAN FIX HEADERS)
# ==========================================

def get_data_from_indodax(coin_symbol, days_back=365):
    """Menarik data historis harian langsung dari API Publik Indodax (Rupiah - IDR)"""
    try:
        end_time = int(datetime.datetime.now().timestamp())
        start_time = end_time - (days_back * 24 * 60 * 60)
        
        url = f"https://indodax.com/tradingview/history?symbol={coin_symbol.lower()}_idr&resolution=D&from={start_time}&to={end_time}"
        
        # FIX: Tambahkan User-Agent agar tidak diblokir oleh Cloudflare/Sistem Indodax
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            st.error(f"Indodax API Return HTTP Status: {response.status_code}")
            return pd.DataFrame()
            
        json_data = response.json()
        
        if json_data.get('s') == 'ok':
            df = pd.DataFrame({
                'Date': pd.to_datetime(json_data['t'], unit='s'),
                'Open': json_data['o'],
                'High': json_data['h'],
                'Low': json_data['l'],
                'Close': json_data['c'],
                'Volume': json_data['v']
            })
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            return df
        else:
            st.error(f"Indodax API merespons dengan error: {json_data.get('errmsg', 'Unknown')}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Gagal memuat data dari Indodax: {e}")
        return pd.DataFrame()

def get_data_from_yahoo(ticker_name, days_back=365):
    """Menarik data historis harian langsung dari Yahoo Finance API (USD)"""
    try:
        hari_ini = datetime.date.today()
        tanggal_mulai = hari_ini - datetime.timedelta(days=days_back)
        
        df = yf.download(
            ticker_name, 
            start=tanggal_mulai.strftime("%Y-%m-%d"), 
            end=hari_ini.strftime("%Y-%m-%d"),
            progress=False
        )
        
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
# 2. ANTARMUKA PENGGUNA (SIDEBAR PARAMETER)
# ==========================================

st.sidebar.title("🛠️ Konfigurasi Sistem")

st.sidebar.subheader("1. Pengaturan Data Source")
sumber_data = st.sidebar.selectbox(
    "Pilih Bursa/Sumber Data:",
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
    koin_terpilih = st.sidebar.selectbox("Pilih Koin Crypto:", list_koin)
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
    koin_terpilih = st.sidebar.selectbox("Pihal Koin Crypto (Pair IDR):", list_koin)

rentang_hari = st.sidebar.slider("Rentang Data Historis (Hari Ke Belakang):", 180, 730, 365)

st.sidebar.subheader("2. Parameter Model ARIMA")
p = st.sidebar.number_input("Orde AR (p):", min_value=0, max_value=5, value=1)
d = st.sidebar.number_input("Differencing (d):", min_value=0, max_value=2, value=1)
q = st.sidebar.number_input("Orde MA (q):", min_value=0, max_value=5, value=1)

st.sidebar.subheader("3. Parameter Model SVR")
kernel = st.sidebar.selectbox("Kernel SVR:", ["rbf", "linear", "poly"])
c_param = st.sidebar.number_input("Regularisasi C:", min_value=1.0, max_value=1000.0, value=100.0)
gamma = st.sidebar.selectbox("Gamma SVR:", ["scale", "auto"])

st.sidebar.subheader("4. Proyeksi Masa Depan")
hari_prediksi = st.sidebar.slider("Jumlah Hari ke Depan:", 1, 30, 7)

# ==========================================
# 3. MAIN PAGE / HALAMAN UTAMA
# ==========================================

st.title("📊 Sistem Analisis & Prediksi Cryptocurrency")
st.markdown("Integrasi Model Linier **ARIMA** dan Machine Learning **SVR (Support Vector Regression)** untuk koreksi residu volatil.")

# Ambil data secara otomatis saat koin/bursa berubah
if sumber_data == "Yahoo Finance (USD)":
    df_crypto = get_data_from_yahoo(koin_terpilih, days_back=rentang_hari)
    mata_uang = "USD"
else:
    df_crypto = get_data_from_indodax(koin_terpilih, days_back=rentang_hari)
    mata_uang = "IDR"

if not df_crypto.empty:
    # ------------------------------------------
    # TAMPILKAN 10 DATA TERBARU & TOMBOL UNDUH
    # ------------------------------------------
    st.subheader(f"📋 10 Data Historis Terbaru ({koin_terpilih} - {mata_uang})")
    st.dataframe(df_crypto.tail(10))
    
    # Fitur Download Dataset Aktual
    csv_data = df_crypto.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Unduh Dataset Aktual (CSV)",
        data=csv_data,
        file_name=f"data_aktual_{koin_terpilih}_{datetime.date.today()}.csv",
        mime="text/csv"
    )
    
    # ------------------------------------------
    # ANALISIS STATISTIK DESKRIPTIF (Statistical Analysis)
    # ------------------------------------------
    st.subheader("📉 Analisis Statistik Deskriptif Dataset")
    col_stat1, col_stat2 = st.columns([2, 1])
    with col_stat1:
        st.dataframe(df_crypto.describe().T)
    with col_stat2:
        st.info(
            f"**Informasi Dataset:**\n\n"
            f"- Tanggal Mulai: {df_crypto.index.min().strftime('%Y-%m-%d')}\n"
            f"- Tanggal Akhir: {df_crypto.index.max().strftime('%Y-%m-%d')}\n"
            f"- Total Sampel: {len(df_crypto)} Hari"
        )

    # ------------------------------------------
    # TOMBOL PROSES PREDIKSI & PEMODELAN HYBRID
    # ------------------------------------------
    st.markdown("---")
    st.subheader("🔮 Pemodelan Komputasi Hybrid")
    st.info("Klik tombol di bawah ini untuk memulai proses pelatihan data menggunakan ARIMA, SVR, dan metode penggabungan Hybrid.")
    
    if st.button("🚀 Jalankan Analisis & Prediksi Model", type="primary"):
        try:
            data_close = df_crypto['Close'].astype(float)
            
            # 1. PROSES MODEL ARIMA
            model_arima = ARIMA(data_close, order=(p, d, q))
            model_arima_fitted = model_arima.fit()
            prediksi_arima = model_arima_fitted.fittedvalues
            residu_arima = data_close - prediksi_arima
            
            # 2. PROSES MODEL SVR (Berdasarkan Lag Residu ARIMA)
            df_residu = pd.DataFrame({'Residu': residu_arima})
            df_residu['Lag_1'] = df_residu['Residu'].shift(1)
            df_residu.dropna(inplace=True)
            
            X_svr = df_residu[['Lag_1']].values
            y_svr = df_residu['Residu'].values
            
            scaler = MinMaxScaler(feature_range=(-1, 1))
            X_svr_scaled = scaler.fit_transform(X_svr)
            
            model_svr = SVR(kernel=kernel, C=c_param, gamma=gamma)
            model_svr.fit(X_svr_scaled, y_svr)
            prediksi_residu_svr = model_svr.predict(X_svr_scaled)
            
            # 3. SINKRONISASI DATAFRAME EVALUASI (Bypass Indeks Pertama karena Lag)
            data_aktual_eval = data_close.iloc[1:]
            prediksi_arima_eval = prediksi_arima.iloc[1:]
            
            # Rekonstruksi hasil akhir SVR murni pada harga asli
            prediksi_svr_eval = prediksi_arima_eval + prediksi_residu_svr 
            hasil_hybrid_eval = prediksi_arima_eval + prediksi_residu_svr
            
            # ------------------------------------------
            # PERHITUNGAN TABEL PERBANDINGAN ERROR (RMSE, MAE, MAPE)
            # ------------------------------------------
            st.subheader("📊 Tabel Perbandingan Nilai Eror Model")
            
            metrics_dict = {
                "Model Pemodelan": ["ARIMA Murni (Linear)", "SVR Murni (Koreksi Residu)", "Hybrid ARIMA-SVR"],
                "RMSE": [
                    root_mean_squared_error(data_aktual_eval, prediksi_arima_eval),
                    root_mean_squared_error(data_aktual_eval, prediksi_svr_eval),
                    root_mean_squared_error(data_aktual_eval, hasil_hybrid_eval)
                ],
                "MAE": [
                    mean_absolute_error(data_aktual_eval, prediksi_arima_eval),
                    mean_absolute_error(data_aktual_eval, prediksi_svr_eval),
                    mean_absolute_error(data_aktual_eval, hasil_hybrid_eval)
                ],
                "MAPE (%)": [
                    mean_absolute_percentage_error(data_aktual_eval, prediksi_arima_eval) * 100,
                    mean_absolute_percentage_error(data_aktual_eval, prediksi_svr_eval) * 100,
                    mean_absolute_percentage_error(data_aktual_eval, hasil_hybrid_eval) * 100
                ]
            }
            
            df_metrics = pd.DataFrame(metrics_dict)
            st.table(df_metrics)
            
            # ------------------------------------------
            # VISUALISASI MODEL COMPARISON
            # ------------------------------------------
            st.subheader("📈 Visualisasi Hasil Prediksi Data Historis")
            
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(data_close.index, data_close.values, label="Harga Aktual Pasar", color="black", linewidth=2)
            ax.plot(prediksi_arima_eval.index, prediksi_arima_eval.values, label="Prediksi ARIMA", color="orange", linestyle="--", alpha=0.8)
            ax.plot(data_aktual_eval.index, hasil_hybrid_eval, label="Prediksi Hybrid ARIMA-SVR", color="crimson", linewidth=2)
            
            ax.set_title(f"Komparasi Pergerakan Harga Aktual vs Model Estimasi ({koin_terpilih})", fontsize=14)
            ax.set_xlabel("Tanggal", fontsize=12)
            ax.set_ylabel(f"Harga ({mata_uang})", fontsize=12)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="upper left")
            
            if mata_uang == "IDR":
                ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
                
            st.pyplot(fig)
            
            # ------------------------------------------
            # PROYEKSI PREDIKSI MASA DEPAN
            # ------------------------------------------
            st.markdown("---")
            st.subheader(f"🔮 Hasil Proyeksi Nilai Masa Depan ({hari_prediksi} Hari Kedepan)")
            
            # Forecasting Masa Depan ARIMA
            forecast_arima_raw = model_arima_fitted.forecast(steps=hari_prediksi)
            
            # Forecasting Masa Depan Residu SVR secara Autoregresif
            residu_terakhir = y_svr[-1]
            forecast_residu_svr_list = []
            
            for _ in range(hari_prediksi):
                scaled_lag = scaler.transform([[residu_terakhir]])
                pred_residu = model_svr.predict(scaled_lag)[0]
                forecast_residu_svr_list.append(pred_residu)
                residu_terakhir = pred_residu
                
            # Kombinasikan Proyeksi Masa Depan
            proyeksi_hybrid_final = forecast_arima_raw + np.array(forecast_residu_svr_list)
            
            # Membuat Indeks Tanggal Masa Depan
            tanggal_akhir_pasar = data_close.index[-1]
            indeks_masa_depan = pd.date_range(start=tanggal_akhir_pasar + datetime.timedelta(days=1), periods=hari_prediksi)
            
            # Dataframe Hasil Prediksi
            df_proyeksi_tabel = pd.DataFrame({
                "Proyeksi Nilai ARIMA Murni": forecast_arima_raw.values,
                "Koreksi Residu SVR": forecast_residu_svr_list,
                f"Hasil Akhir Hybrid ({mata_uang})": proyeksi_hybrid_final
            }, index=indeks_masa_depan)
            df_proyeksi_tabel.index.name = "Tanggal Proyeksi"
            
            # Tampilkan Komparasi Output Prediksi dalam Bentuk Kolom (Tabel & Grafik)
            col_res1, col_res2 = st.columns([1, 1])
            with col_res1:
                st.markdown("**Tabel Rincian Angka Prediksi:**")
                st.dataframe(df_proyeksi_tabel)
            with col_res2:
                st.markdown("**Grafik Tren Proyeksi Masa Depan:**")
                fig_f, ax_f = plt.subplots(figsize=(10, 5))
                ax_f.plot(df_proyeksi_tabel.index, df_proyeksi_tabel[f"Hasil Akhir Hybrid ({mata_uang})"], marker='o', color='crimson', label='Proyeksi Hybrid')
                ax_f.plot(df_proyeksi_tabel.index, df_proyeksi_tabel["Proyeksi Nilai ARIMA Murni"], marker='x', linestyle='--', color='orange', label='Proyeksi ARIMA')
                ax_f.set_ylabel(f"Harga ({mata_uang})")
                ax_f.grid(True, linestyle=":", alpha=0.5)
                ax_f.legend()
                plt.xticks(rotation=45)
                st.pyplot(fig_f)
                
        except Exception as math_err:
            st.error(f"Gagal memproses perhitungan model: {math_err}")
            st.info("Saran: Sesuaikan kembali nilai parameter ordo p, d, q ARIMA pada sidebar agar data mencapai titik stasioner yang ideal.")
            
else:
    st.warning("Saluran data kosong. Silakan periksa kembali koneksi jaringan atau ganti opsi bursa pilihan Anda.")
