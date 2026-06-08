import streamlit as pd
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
        end_time = int(datetime.datetime.now().timestamp())
        start_time = end_time - (days_back * 24 * 60 * 60)
        
        # Endpoint resmi TradingView History Indodax (Resolution 'D' = Harian)
        url = f"https://indodax.com/tradingview/history?symbol={coin_symbol.lower()}_idr&resolution=D&from={start_time}&to={end_time}"
        
        response = requests.get(url, timeout=10)
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
            # Normalisasi multi-index jika library yfinance versi terbaru mengembalikannya
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
                
            df.index = pd.to_datetime(df.index)
            df.index.name = 'Date'
            df.sort_index(inplace=True)
            
            # Memastikan kolom bertipe numerik float
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

# --- OPSI 1: PILIHAN SUMBER DATA ---
st.sidebar.subheader("1. Sumber Data & Koin")
sumber_data = st.sidebar.selectbox(
    "Pilih Sumber Data:",
    ["Yahoo Finance (USD)", "Indodax Bursa Indonesia (IDR)"]
)

# Sesuaikan list koin berdasarkan pilihan sumber data agar user tidak bingung format
if sumber_data == "Yahoo Finance (USD)":
    list_koin = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD"]
    koin_terpilih = st.sidebar.selectbox("Pilih Koin Crypto:", list_koin)
else:
    list_koin = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA"]
    koin_terpilih = st.sidebar.selectbox("Pilih Koin Crypto (Pair IDR):", list_koin)

# Parameter Rentang Waktu Historis
rentang_hari = st.sidebar.slider("Rentang Data Historis (Hari ke Belakang):", 180, 730, 365)

# --- OPSI 2: PARAMETER ARIMA ---
st.sidebar.subheader("2. Parameter ARIMA (Linear)")
p = st.sidebar.number_input("Orde AR (p):", min_value=0, max_value=5, value=1, step=1)
d = st.sidebar.number_input("Differencing (d):", min_value=0, max_value=2, value=1, step=1)
q = st.sidebar.number_input("Orde MA (q):", min_value=0, max_value=5, value=1, step=1)

# --- OPSI 3: PARAMETER SVR ---
st.sidebar.subheader("3. Parameter SVR (Non-Linear Residu)")
kernel = st.sidebar.selectbox("Kernel SVR:", ["rbf", "linear", "poly"])
c_param = st.sidebar.number_input("Nilai Regularisasi C:", min_value=0.1, max_value=1000.0, value=100.0, step=10.0)
gamma = st.sidebar.selectbox("Gamma SVR:", ["scale", "auto"])

# Parameter Prediksi Masa Depan
hari_prediksi = st.sidebar.slider("Jumlah Hari Prediksi ke Depan:", 1, 30, 7)

# ==========================================
# 3. ALUR UTAMA EKSEKUSI DATA (MAIN PAGE)
# ==========================================

st.title("📊 Aplikasi Prediksi Cryptocurrency Berbasis Hybrid ARIMA-SVR")
st.markdown("Aplikasi ini mengombinasikan model klasik **ARIMA** (untuk menangkap tren linear) dan Machine Learning **SVR** (untuk memodelkan residu/galat non-linear).")

# Mengambil data berdasarkan opsi yang dipilih pengguna
with st.spinner("Mengunduh data secara real-time dari API bursa..."):
    if sumber_data == "Yahoo Finance (USD)":
        df_crypto = get_data_from_yahoo(koin_terpilih, days_back=rentang_hari)
        mata_uang = "USD"
    else:
        df_crypto = get_data_from_indodax(koin_terpilih, days_back=rentang_hari)
        mata_uang = "IDR"

# Validasi jika data berhasil diambil
if not df_crypto.empty:
    st.success(f"Berhasil memuat {len(df_crypto)} baris data historis {koin_terpilih} dari {sumber_data}!")
    
    # Tampilkan Ringkasan Data Historis dalam Expander
    with st.expander("👀 Lihat Tabel Data Historis Lengkap"):
        st.dataframe(df_crypto.tail(10))
        
    # Ambil data harga 'Close' sebagai target analisis runtun waktu
    data_close = df_crypto['Close'].astype(float)
    
    # ==========================================
    # 4. PEMODELAN HYBRID (ARIMA + SVR)
    # ==========================================
    
    try:
        # ---- LANGKAH 1: Pemodelan Linear dengan ARIMA ----
        # Fit model ARIMA berdasarkan nilai p, d, q dari input pengguna
        model_arima = ARIMA(data_close, order=(p, d, q))
        model_arima_fitted = model_arima.fit()
        
        # Dapatkan nilai hasil prediksi ARIMA di masa lalu (In-sample prediction)
        prediksi_linear_arima = model_arima_fitted.fittedvalues
        
        # Hitung Nilai Residu/Galat (Data Aktual dikurangi Prediksi Linear ARIMA)
        residu_arima = data_close - prediksi_linear_arima
        
        # ---- LANGKAH 2: Pemodelan Non-Linear Residu dengan SVR ----
        # Siapkan struktur data lagged untuk melatih SVR menangkap pola residu
        df_residu = pd.DataFrame({'Residu': residu_arima})
        df_residu['Lag_1'] = df_residu['Residu'].shift(1)
        df_residu.dropna(inplace=True)
        
        X_svr = df_residu[['Lag_1']].values
        y_svr = df_residu['Residu'].values
        
        # Skalasi Fitur Residu demi performa maksimal kernel SVR
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_svr_scaled = scaler.fit_transform(X_svr)
        
        # Inisialisasi dan latih model SVR
        model_svr = SVR(kernel=kernel, C=c_param, gamma=gamma)
        model_svr.fit(X_svr_scaled, y_svr)
        
        # Prediksi pola residu non-linear
        prediksi_residu_svr = model_svr.predict(X_svr_scaled)
        
        # ---- LANGKAH 3: Kombinasi Hybrid (Hasil ARIMA + Hasil SVR) ----
        # Sesuaikan panjang baris data karena efek lag_1 pada proses SVR
        data_aktual_cocok = data_close.iloc[1:]
        prediksi_arima_cocok = prediksi_linear_arima.iloc[1:]
        
        # Nilai akhir Hybrid = Hasil ARIMA + Hasil SVR koreksi residu
        hasil_prediksi_hybrid = prediksi_arima_cocok + prediksi_residu_svr
        
        # ==========================================
        # 5. METRIK EVALUASI AKURASI MODEL
        # ==========================================
        
        # Hitung Metrik Akurasi Nilai Hybrid dibanding Data Aktual
        mape_hybrid = mean_absolute_percentage_error(data_aktual_cocok, hasil_prediksi_hybrid) * 100
        rmse_hybrid = root_mean_squared_error(data_aktual_cocok, hasil_prediksi_hybrid)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="📊 Akurasi Model (MAPE)", value=f"{mape_hybrid:.2f} %")
        with col2:
            st.metric(label="📉 Nilai Eror (RMSE)", value=f"{rmse_hybrid:,.2f} {mata_uang}")
            
        # ==========================================
        # 6. VISUALISASI GRAFIK HASIL
        # ==========================================
        
        st.subheader(f"📈 Grafik Perbandingan Harga Aktual vs Hasil Prediksi Hybrid ({koin_terpilih})")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(data_close.index, data_close.values, label="Data Aktual Asli (Pasar)", color="royalblue", linewidth=2)
        ax.plot(data_aktual_cocok.index, hasil_prediksi_hybrid, label="Hasil Model Hybrid ARIMA-SVR", color="crimson", linestyle="--", linewidth=1.5)
        
        ax.set_title(f"Analisis Tren Finansial {koin_terpilih} ({sumber_data})", fontsize=14)
        ax.set_xlabel("Tanggal", fontsize=12)
        ax.set_ylabel(f"Harga ({mata_uang})", fontsize=12)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left")
        
        if mata_uang == "IDR":
            ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
            
        st.pyplot(fig)
        
        # ---- LANGKAH 4: PROYEKSI PREDIKSI MASA DEPAN ----
        st.subheader(f"🔮 Proyeksi Prediksi Masa Depan ({hari_prediksi} Hari ke Depan)")
        
        # 1. Forecast Masa Depan dari ARIMA
        forecast_arima = model_arima_fitted.forecast(steps=hari_prediksi)
        
        # 2. Forecast Residu dari SVR untuk masa depan secara autoregresif
        residu_terakhir = y_svr[-1]
        forecast_residu_svr = []
        
        for _ in range(hari_prediksi):
            scaled_lag = scaler.transform([[residu_terakhir]])
            pred_residu = model_svr.predict(scaled_lag)[0]
            forecast_residu_svr.append(pred_residu)
            residu_terakhir = pred_residu  # Update lag dengan hasil prediksi terbaru
            
        # 3. Gabungkan nilai Proyeksi Masa Depan
        proyeksi_masa_depan_hybrid = forecast_arima + np.array(forecast_residu_svr)
        
        # Buat Tanggal indeks untuk masa depan
        tanggal_terakhir = data_close.index[-1]
        tanggal_masa_depan = pd.date_range(start=tanggal_terakhir + datetime.timedelta(days=1), periods=hari_prediksi)
        
        df_proyeksi = pd.DataFrame({
            f"Proyeksi Harga Hybrid ({mata_uang})": proyeksi_masa_depan_hybrid
        }, index=tanggal_masa_depan)
        df_proyeksi.index.name = "Tanggal Masa Depan"
        
        st.table(df_proyeksi)
        
    except Exception as model_error:
        st.error(f"Gagal melakukan komputasi matriks matematika model Hybrid: {model_error}")
        st.info("Saran: Coba sesuaikan kombinasi nilai p, d, q ARIMA Anda pada sidebar untuk kecocokan stasioneritas data.")
else:
    st.warning("Gagal memproses visualisasi karena pipa penarikan data kosong. Silakan periksa kembali koneksi internet Anda.")
