import streamlit as st
import pandas as pd
import pmdarima as pm

# 1. Konfigurasi Halaman Antarmuka Streamlit
st.set_page_config(page_title="Prediksi Saham ARIMA", layout="centered")
st.title("🚀 Aplikasi Prediksi Saham (Auto-ARIMA)")
st.write("Masukkan data historis untuk melakukan forecasting otomatis.")

# 2. Input Data dari Pengguna di Antarmuka Web
# Memberikan nilai default berupa list angka agar user langsung melihat contohnya
input_data = st.text_area(
    "Masukkan data historis (pisahkan dengan koma):", 
    value="100.0, 101.5, 102.3, 103.1, 104.5, 105.2, 106.0, 107.8, 108.5, 109.1, 110.0"
)

forecast_horizon = st.number_input("Jumlah Langkah Prediksi ke Depan (Horizon):", min_value=1, max_value=30, value=5)

# 3. Tombol Eksekusi Prediksi
if st.button("Jalankan Prediksi"):
    try:
        # Proses parsing input teks menjadi list dari angka float
        historical_prices = [float(x.strip()) for x in input_data.split(",") if x.strip()]
        
        if len(historical_prices) < 5:
            st.error("Data terlalu sedikit! Masukkan minimal 5 data historis.")
        else:
            with st.spinner("Sedang menghitung model Auto-ARIMA..."):
                # Konversi ke pandas Series
                series = pd.Series(historical_prices)
                
                # Training model ARIMA otomatis
                model = pm.auto_arima(series, seasonal=False, stepwise=True, suppress_warnings=True, error_action="ignore")
                
                # Melakukan forecasting
                predictions = model.predict(n_periods=int(forecast_horizon)).tolist()
                
            # Tampilkan Hasil di Layar Web Streamlit
            st.success("Prediksi Berhasil!")
            
            # Tampilkan dalam bentuk dataframe/tabel ringkas
            df_res = pd.DataFrame({
                "Langkah ke-": [i+1 for i in range(len(predictions))],
                "Hasil Prediksi": predictions
            })
            st.dataframe(df_res, use_container_width=True)
            
            # Opsional: Tampilkan grafik garis sederhana
            st.line_chart(predictions)
            
    except ValueError:
        st.error("Format data salah! Pastikan hanya memasukkan angka yang dipisahkan oleh koma (contoh: 10.5, 11.2, 13.0).")
    except Exception as e:
        st.error(f"Terjadi kesalahan pada model: {e}")
