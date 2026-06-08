import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
import matplotlib.pyplot as plt

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Prediksi Crypto Hybrid ARIMA-SVR", layout="wide")

# ==========================================
# 1. FUNGSI PENARIKAN DATA (DATA PIPELINE)
# ==========================================

def get_data_from_indodax(coin_symbol, days_back=365):
    """Menarik data historis harian langsung dari API Publik Indodax (Rupiah - IDR)"""
    try:
        # Indodax TradingView API menggunakan UNIX Timestamp dalam satuan DETIK
        end_time = int(datetime.datetime.now().timestamp())
        start_time = end_time - (days_back * 24 * 60 * 60)
        
        # Endpoint publik resmi Crypto Bursa Indonesia (Indodax)
        url = f"https://indodax.com/tradingview/history?symbol={coin_symbol.lower()}_idr&resolution=D&from={start_time}&to={end_time}"
        
        # Tambahkan headers agar tidak diblokir oleh firewall bursa
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            st.error(f"API Indodax offline. Status Code: {response.status_code}")
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
            df.sort_index(inplace=True) # Urutkan dari tanggal lampau ke tanggal terbaru
            
            # Memastikan seluruh kolom bertipe data numerik float
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        else:
            st.error(f"Respons pasar Indodax bermasalah: {json_data.get('errmsg', 'No OK Status')}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Gagal memuat data dari Indodax: {e}")
        return pd.DataFrame()

def get_data_from_yahoo(ticker_name, days_back=365):
    """Menarik data historis harian langsung dari Yahoo Finance API via yfinance (USD)"""
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

st.sidebar.title("🛠️ Konfigurasi Aplikasi")

st.sidebar.subheader("1. Sumber Data & Koin")
sumber_data = st.sidebar.selectbox(
    "Pilih Bursa / Sumber Data:",
    ["Indodax Bursa Indonesia (IDR)", "Yahoo Finance (USD)"]
)

if sumber_data == "Yahoo Finance (USD)":
    list_koin = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD"]
    koin_terpilih = st.sidebar.selectbox("Pilih Koin Crypto:", list_koin)
else:
    list_koin = ["BTC", "ETH", "BNB", "SOL", "XRP"]
    koin_terpilih = st.sidebar.selectbox("Pilih Koin Crypto (Pair IDR):", list_koin)

rentang_hari = st.sidebar.slider("Rentang Data Historis (Hari ke Belakang):", 180, 730, 365)

st.sidebar.subheader("2. Parameter ARIMA (Linear)")
p = st.sidebar.number_input("Orde AR (p):", min_value=0, max_value=5, value=1, step=1)
d = st.sidebar.number_input("Differencing (d):", min_value=0, max_value=2, value=1, step=1)
q = st.sidebar.number_input("Orde MA (q):", min_value=0, max_value=5, value=1, step=1)

st.sidebar.subheader("3. Parameter SVR (Non-Linear Residu)")
kernel = st.sidebar.selectbox("Kernel SVR:", ["rbf", "linear", "poly"])
c_param = st.sidebar.number_input("Nilai Regularisasi C:", min_value=0.1, max_value=1000.0, value=100.0, step=10.0)
gamma = st.sidebar.selectbox("Gamma SVR:", ["scale", "auto"])

st.sidebar.subheader("4. Pengaturan Masa Depan")
hari_prediksi = st.sidebar.slider("Jumlah Hari Prediksi ke Depan:", 1, 30, 7)

# ==========================================
# 3. ALUR UTAMA HALAMAN UTAMA (MAIN PAGE)
# ==========================================

st.title("📊 Aplikasi Prediksi Cryptocurrency Berbasis Hybrid ARIMA-SVR")
st.markdown("Aplikasi analisis cerdas yang mengombinasikan model statistik linear **ARIMA** dan optimasi residu non-linear **SVR**.")

# Memulai proses penarikan data dari API bursa terpilih
with st.spinner("Mengunduh data runtun waktu pasar..."):
    if sumber_data == "Yahoo Finance (USD)":
        df_crypto = get_data_from_yahoo(koin_terpilih, days_back=rentang_hari)
        mata_uang = "USD"
    else:
        df_crypto = get_data_from_indodax(koin_terpilih, days_back=rentang_hari)
        mata_uang = "IDR"

# Validasi Keberadaan Dataset
if not df_crypto.empty:
    st.success(f"Berhasil mengamankan data aktual {koin_terpilih} dari bursa {sumber_data}!")
    
    # --- FITUR 1: MENAMPILKAN TABEL 10 BARIS DATA TERBARU ---
    st.subheader(f"📋 Tabel 10 Baris Data Terbaru {koin_terpilih}")
    
    # Ambil 10 baris terakhir, urutkan agar tanggal paling baru berada di paling atas tabel
    df_tampil = df_crypto.tail(10).copy()
    df_tampil.index = df_tampil.index.strftime('%Y-%m-%d')
    st.dataframe(df_tampil.iloc[::-1], use_container_width=True)
    
    # --- FITUR 2: TOMBOL UNDUH DATASET ASLI ---
    csv_data = df_crypto.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Unduh Seluruh Dataset Historis (.CSV)",
        data=csv_data,
        file_name=f"dataset_historis_{koin_terpilih.lower()}_{mata_uang.lower()}.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # --- FITUR 3: TOMBOL PREDIKSI DAN ANALISIS MODEL HYBRID ---
    st.subheader("🤖 Ruang Komputasi Matematika Model")
    st.info("Klik tombol di bawah ini untuk memulai pelatihan algoritma ARIMA, ekstraksi residu, optimasi SVR, serta peramalan masa depan.")
    
    tombol_prediksi = st.button("🚀 Jalankan Analisis Prediksi Model Hybrid")
    
    if tombol_prediksi:
        try:
            with st.spinner("Algoritma sedang mempelajari pola tren dan volatilitas harga..."):
                
                # Menyiapkan data target analisis
                data_close = df_crypto['Close'].astype(float)
                
                # ---- PEMODELAN INDEPENDEN 1: MURNI ARIMA ----
                model_arima = ARIMA(data_close, order=(p, d, q))
                model_arima_fitted = model_arima.fit()
                prediksi_linear_arima = model_arima_fitted.fittedvalues
                
                # Hitung Nilai Residu (Galat)
                residu_arima = data_close - prediksi_linear_arima
                
                # ---- PEMODELAN INDEPENDEN 2: MURNI SVR ----
                # Menggunakan lag 1 hari dari harga Close asli untuk melatih SVR Mandiri
                df_svr_tunggal = pd.DataFrame({'Close': data_close})
                df_svr_tunggal['Lag_1'] = df_svr_tunggal['Close'].shift(1)
                df_svr_tunggal.dropna(inplace=True)
                
                X_single = df_svr_tunggal[['Lag_1']].values
                y_single = df_svr_tunggal['Close'].values
                
                scaler_single = MinMaxScaler(feature_range=(-1, 1))
                X_single_scaled = scaler_single.fit_transform(X_single)
                
                model_svr_tunggal = SVR(kernel=kernel, C=c_param, gamma=gamma)
                model_svr_tunggal.fit(X_single_scaled, y_single)
                prediksi_murni_svr = model_svr_tunggal.predict(X_single_scaled)
                
                # ---- PEMODELAN GABUNGAN 3: HYBRID ARIMA-SVR ----
                df_residu = pd.DataFrame({'Residu': residu_arima})
                df_residu['Lag_1'] = df_residu['Residu'].shift(1)
                df_residu.dropna(inplace=True)
                
                X_hybrid = df_residu[['Lag_1']].values
                y_hybrid = df_residu['Residu'].values
                
                scaler_hybrid = MinMaxScaler(feature_range=(-1, 1))
                X_hybrid_scaled = scaler_hybrid.fit_transform(X_hybrid)
                
                model_svr_residu = SVR(kernel=kernel, C=c_param, gamma=gamma)
                model_svr_residu.fit(X_hybrid_scaled, y_hybrid)
                prediksi_residu_svr = model_svr_residu.predict(X_hybrid_scaled)
                
                # Kombinasikan hasil ARIMA dan koreksi eror SVR
                data_aktual_cocok = data_close.iloc[1:]
                prediksi_arima_cocok = prediksi_linear_arima.iloc[1:]
                hasil_prediksi_hybrid = prediksi_arima_cocok + prediksi_residu_svr
                
                # ==========================================
                # 4. TABEL PERBANDINGAN ERROR MASING-MASING MODEL
                # ==========================================
                st.subheader("🎯 Tabel Perbandingan Performa Akurasi Model")
                
                # Hitung MAPE & RMSE untuk Murni ARIMA
                mape_arima = mean_absolute_percentage_error(data_close, prediksi_linear_arima) * 100
                rmse_arima = root_mean_squared_error(data_close, prediksi_linear_arima)
                
                # Hitung MAPE & RMSE untuk Murni SVR
                mape_svr = mean_absolute_percentage_error(y_single, prediksi_murni_svr) * 100
                rmse_svr = root_mean_squared_error(y_single, prediksi_murni_svr)
                
                # Hitung MAPE & RMSE untuk Model Kombinasi Hybrid
                mape_hybrid = mean_absolute_percentage_error(data_aktual_cocok, hasil_prediksi_hybrid) * 100
                rmse_hybrid = root_mean_squared_error(data_aktual_cocok, hasil_prediksi_hybrid)
                
                # Susun data perbandingan ke dalam Dataframe ringkas
                df_error_komparasi = pd.DataFrame({
                    "Model Algoritma": ["Murni ARIMA (Linear)", "Murni SVR (Non-Linear)", "Hybrid ARIMA-SVR (Saran)"],
                    "MAPE (%)": [f"{mape_arima:.3f} %", f"{mape_svr:.3f} %", f"{mape_hybrid:.3f} %"],
                    "RMSE": [f"{rmse_arima:,.2f} {mata_uang}", f"{rmse_svr:,.2f} {mata_uang}", f"{rmse_hybrid:,.2f} {mata_uang}"]
                })
                st.table(df_error_komparasi.set_index("Model Algoritma"))
                
                # ==========================================
                # 5. GRAFIK ANALISIS PREDIKSI DATA AKTUAL
                # ==========================================
                st.subheader("📈 Visualisasi Hasil Fit Model vs Pasar Aktual")
                
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(data_close.index, data_close.values, label="Harga Aktual Pasar", color="royalblue", linewidth=2)
                ax.plot(data_aktual_cocok.index, hasil_prediksi_hybrid, label="Hasil Prediksi Koreksi Hybrid", color="crimson", linestyle="--", linewidth=1.5)
                
                ax.set_title(f"Grafik Hasil Pelatihan Model Hybrid {koin_terpilih}", fontsize=14)
                ax.set_xlabel("Tanggal Analisis", fontsize=11)
                ax.set_ylabel(f"Nilai Aset ({mata_uang})", fontsize=11)
                ax.grid(True, linestyle=":", alpha=0.5)
                ax.legend(loc="upper left")
                
                if mata_uang == "IDR":
                    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
                st.pyplot(fig)
                
                # ==========================================
                # 6. PROYEKSI PREDIKSI MASA DEPAN + TABEL HASIL
                # ==========================================
                st.subheader(f"🔮 Hasil Proyeksi Nilai Masa Depan ({hari_prediksi} Hari ke Depan)")
                
                # Proyeksi Linear ARIMA
                forecast_arima = model_arima_fitted.forecast(steps=hari_prediksi)
                
                # Proyeksi Residu SVR secara Autoregresif
                residu_terakhir = y_hybrid[-1]
                forecast_residu_svr = []
                for _ in range(hari_prediksi):
                    scaled_lag = scaler_hybrid.transform([[residu_terakhir]])
                    pred_residu = model_svr_residu.predict(scaled_lag)[0]
                    forecast_residu_svr.append(pred_residu)
                    residu_terakhir = pred_residu
                
                # Gabungkan Proyeksi Nilai Akhir
                proyeksi_hybrid = forecast_arima + np.array(forecast_residu_svr)
                
                # Membuat Indeks Tanggal Masa Depan yang Berkelanjutan
                tanggal_terakhir = data_close.index[-1]
                tanggal_proyeksi = pd.date_range(start=tanggal_terakhir + datetime.timedelta(days=1), periods=hari_prediksi)
                
                # Tampilkan ke dalam Tabel Aplikasi
                df_tabel_depan = pd.DataFrame({
                    "Estimasi Tanggal": tanggal_proyeksi.strftime('%Y-%m-%d'),
                    f"Prediksi Nilai Harga ({mata_uang})": [f"{x:,.2f}" for x in proyeksi_hybrid]
                })
                
                col_tabel, col_grafik_depan = st.columns([2, 3])
                
                with col_tabel:
                    st.markdown("**Tabel Angka Estimasi Masa Depan:**")
                    st.dataframe(df_tabel_depan.set_index("Estimasi Tanggal"), use_container_width=True)
                    
                with col_grafik_depan:
                    st.markdown("**Grafik Garis Tren ke Depan:**")
                    fig_depan, ax_depan = plt.subplots(figsize=(10, 6))
                    ax_depan.plot(tanggal_proyeksi, proyeksi_hybrid, marker='o', linestyle='-', color='darkorange', linewidth=2, label="Proyeksi Masa Depan")
                    ax_depan.set_ylabel(f"Harga ({mata_uang})")
                    ax_depan.grid(True, linestyle=":", alpha=0.5)
                    plt.xticks(rotation=30)
                    if mata_uang == "IDR":
                        ax_depan.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
                    st.pyplot(fig_depan)
                    
        except Exception as math_err:
            st.error(f"Gagal memproses perhitungan model: {math_err}")
            st.info("Tip Keamanan Data: Ubah nilai orde ARIMA d atau p pada sidebar untuk menyesuaikan stasioneritas koin saat ini.")
else:
    st.warning("Saluran pipa data kosong. Harap periksa koneksi internet Anda atau coba ubah pilihan bursa di sidebar.")
