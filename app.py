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

# Konfigurasi Dasar Tampilan Web App
st.set_page_config(page_title="Sistem Hybrid ARIMA-SVR Cryptocurrency", layout="wide")

# ==========================================
# 1. PIPELINE DATA (ANTI-BLOKIR CLOUDFLARE INDODAX)
# ==========================================

def get_data_from_indodax(coin_symbol, days_back=365):
    """Menarik data historis dengan simulasi Session TLS Browser penuh agar lolos Cloudflare"""
    try:
        end_time = int(datetime.datetime.now().timestamp())
        start_time = end_time - (days_back * 24 * 60 * 60)
        
        pair_ticker = f"{coin_symbol.upper()}_IDR"
        url = f"https://indodax.com/tradingview/history?symbol={pair_ticker}&resolution=D&from={start_time}&to={end_time}"
        
        # Membuat session HTTP untuk mempertahankan cookie/handshake layaknya browser asli
        session = requests.Session()
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
            "Cache-Control": "max-age=0",
            "Sec-Ch-Ua": '"Not-A.Brand";v="99", "Chromium";v="124", "Google Chrome";v="124"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"macOS"',
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1"
        }
        
        response = session.get(url, headers=headers, timeout=15)
        
        # JIKA TETAP DIBLOKIR CLOUDFLARE (Mengembalikan HTML, bukan JSON)
        if response.text.strip().startswith("<!DOCTYPE html>"):
            st.warning("⚠️ Server Indodax mendeteksi skrip Cloud. Mengaktifkan Jalur Cadangan Konversi Kurs Yahoo Finance...")
            # JALUR CADANGAN: Ambil data dari Yahoo Finance (USD) lalu konversi ke IDR secara otomatis
            df_usd = get_data_from_yahoo(f"{coin_symbol.upper()}-USD", days_back)
            if not df_usd.empty:
                # Ambil kurs USD ke IDR terbaru (Default: Rp 16.000 jika API Kurs lambat)
                kurs_usd_idr = 16200.0
                try:
                    forex_data = yf.download("IDR=X", period="1d", progress=False)
                    if not forex_data.empty:
                        kurs_usd_idr = float(forex_data['Close'].iloc[-1])
                except:
                    pass
                
                # Konversi seluruh kolom ke Rupiah
                for col in ['Open', 'High', 'Low', 'Close']:
                    df_usd[col] = df_usd[col] * kurs_usd_idr
                return df_usd
            return pd.DataFrame()
            
        json_data = response.json()
        
        if json_data.get('s') == 'ok' or 't' in json_data:
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
            
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Gagal melakukan pembacaan struktur API Indodax: {e}")
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
# 2. PANEL SIDEBAR (KONFIGURASI PARAMETER)
# ==========================================

st.sidebar.title("⚙️ Pengaturan Parameter")

st.sidebar.subheader("1. Sumber Data Pasar")
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
    koin_terpilih = st.sidebar.selectbox("Pilih Koin Kripto (Pair IDR):", list_koin)

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

st.title("📊 Platform Komputasi Cerdas Hybrid ARIMA-SVR")
st.markdown("Aplikasi analisis integrasi regresi klasik linier **ARIMA** dengan optimasi pola non-linier residu menggunakan **SVR**.")

# Eksekusi fungsi penarikan data secara real-time
with st.spinner("Memuat data transaksi dari server bursa..."):
    if sumber_data == "Yahoo Finance (USD)":
        df_crypto = get_data_from_yahoo(koin_terpilih, days_back=rentang_hari)
        mata_uang = "USD"
    else:
        df_crypto = get_data_from_indodax(koin_terpilih, days_back=rentang_hari)
        mata_uang = "IDR"

if not df_crypto.empty:
    
    # --- BAGIAN A: EKSPLORASI DATA TERBARU & TOMBOL DOWNLOAD ---
    col_tabel, col_unduh = st.columns([3, 1])
    with col_tabel:
        st.subheader(f"📋 10 Baris Data Transaksi Terakhir ({koin_terpilih})")
        st.dataframe(df_crypto.tail(10))
    with col_unduh:
        st.markdown("<br><br>", unsafe_allow_html=True)
        csv_buffer = df_crypto.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Unduh File Dataset (.CSV)",
            data=csv_buffer,
            file_name=f"dataset_aktual_{koin_terpilih}_{datetime.date.today()}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # --- BAGIAN B: ANALISIS STATISTIK DESKRIPTIF ---
    st.subheader("📊 Analisis Statistik Deskriptif Variabel")
    col_stat_df, col_stat_box = st.columns([2, 1])
    with col_stat_df:
        st.dataframe(df_crypto.describe().T)
    with col_stat_box:
        st.markdown(
            f"<div style='padding:15px; border-radius:10px; background-color:#1e293b; color:white'>"
            f"<b>Metadata Observasi:</b><br>"
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
            ax_hist.plot(data_close.index, data_close.values, label="Harga Pasar Aktual", color="darkcyan", linewidth=2.5)
            ax_hist.plot(prediksi_arima_sinkron.index, prediksi_arima_sinkron.values, label="Estimasi ARIMA", color="orange", linestyle="--", alpha=0.8)
            ax_hist.plot(data_aktual_sinkron.index, hasil_prediksi_hybrid, label="Estimasi Hybrid (ARIMA-SVR)", color="crimson", linewidth=2)
            
            ax_hist.set_title(f"Grafik Komparasi Pergerakan Harga vs Estimasi Model Finansial ({koin_terpilih})", fontsize=13)
            ax_hist.set_xlabel("Skala Waktu (Tanggal)", fontsize=11)
            ax_hist.set_ylabel(f"Nilai Penutupan Aset ({mata_uang})", fontsize=11)
            ax_hist.grid(True, linestyle=":", alpha=0.5)
            ax_hist.legend(loc="upper left")
            
            if mata_uang == "IDR":
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
    st.warning("Saluran data kosong. Silakan periksa kembali koneksi internet Anda.")
