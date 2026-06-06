import streamlit as st
import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import datetime

# --- Pilih Saham & Periode ---
TICKER_SYMBOL = "AAPL" 
START_DATE = "2020-01-01" 
END_DATE = datetime.date.today().strftime("%Y-%m-%d") 

st.title(f"Prediksi Harga Saham {TICKER_SYMBOL}")

@st.cache_data 
def get_stock_data(ticker, start, end):
    """Mengambil data historis saham dari Yahoo Finance."""
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data)
    # Sesuaikan nama kolom hasil ekspor Google Sheet Anda
    df['Date'] = pd.to_datetime(df['Date'])
    df['Close'] = pd.to_numeric(df['Close'])
    df.set_index('Date', inplace=True)
    return df

    df_saham = load_data(GAS_URL)
    st.write("### Data Historis Saham (Dari Google Sheets)", df_saham.tail())

    # Visualisasi Data Historis
    st.line_chart(df_saham['Close'])

    # 2. Input Parameter ARIMA oleh Pengguna
    st.sidebar.header("Parameter Model ARIMA")
    p = st.sidebar.number_input("Orde AR (p)", min_value=0, max_value=5, value=1)
    d = st.sidebar.number_input("Differencing (d)", min_value=0, max_value=2, value=1)
    q = st.sidebar.number_input("Orde MA (q)", min_value=0, max_value=5, value=1)
    
    forecast_steps = st.sidebar.slider("Horizon Prediksi (Hari ke Depan)", 1, 30, 7)

    if st.button("Jalankan Prediksi ARIMA"):
        with st.spinner("Melatih model..."):
            # 3. Training Model ARIMA
            model = ARIMA(df_saham['Close'], order=(p, d, q))
            model_fit = model.fit()
            
            # Forecast
            forecast = model_fit.forecast(steps=forecast_steps)
            
            # Membuat index tanggal untuk hasil prediksi
            last_date = df_saham.index[-1]
            forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps)
            df_forecast = pd.DataFrame({'Prediksi Harga': forecast.values}, index=forecast_index)
            
            # Tampilkan Hasil
            st.write(f"### Hasil Prediksi {forecast_steps} Hari ke Depan", df_forecast)
            
            # Plot Gabungan Data Historis & Prediksi
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df_saham['Close'].tail(50), label='Historis (50 Hari Terakhir)')
            ax.plot(df_forecast['Prediksi Harga'], label='Hasil Prediksi ARIMA', color='red', marker='o')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

except Exception as e:
  st.error(f"Gagal memuat data. Pastikan URL GAS benar. Error: {e}")

# --- Tampilkan Input Pengguna untuk Saham dan Tanggal ---
st.sidebar.header("Pengaturan Data")
selected_ticker = st.sidebar.text_input("Simbol Saham", TICKER_SYMBOL).upper()
selected_start_date = st.sidebar.date_input("Tanggal Mulai", datetime.date(2020, 1, 1))
selected_end_date = st.sidebar.date_input("Tanggal Akhir", datetime.date.today())

if selected_end_date < selected_start_date:
    st.sidebar.error("Tanggal akhir harus setelah tanggal mulai.")
    st.stop() 

# --- Ambil Data ---
df_stock = get_stock_data(selected_ticker, selected_start_date.strftime("%Y-%m-%d"), selected_end_date.strftime("%Y-%m-%d"))

if not df_stock.empty:
    st.subheader(f"Data Historis Harga Saham {selected_ticker}")
    st.write(df_stock.tail()) 
    
    # Perbaikan KeyError: Karena auto_adjust=True, gunakan kolom 'Close'
    if 'Close' in df_stock.columns:
        st.line_chart(df_stock['Close']) 
    else:
        # Cadangan jika kolom MultiIndex tetap terbentuk di beberapa versi enviroment
        st.line_chart(df_stock.iloc[:, 0]) 
else:
    st.warning("Tidak ada data saham yang tersedia atau terjadi kesalahan saat mengambil data.")
