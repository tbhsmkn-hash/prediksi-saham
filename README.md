# Crypto Hybrid Prediction System (ARIMA-SVR)

Aplikasi web berbasis Streamlit untuk memprediksi harga aset kripto menggunakan pendekatan model hybrid: **ARIMA** (untuk menangkap pola linear) dan **SVR** (untuk mengoreksi residu/pola non-linear).

## 🚀 Fitur
*   **Data Real-time:** Mengambil data historis dari *Yahoo Finance* (via `yfinance`).
*   **Hybrid Modeling:** Menggabungkan statistik linier ARIMA dengan mesin pembelajaran SVR.
*   **Interactive Dashboard:** Visualisasi pergerakan harga dan perbandingan akurasi model menggunakan *Matplotlib*.
*   **Downloadable Data:** Kemudahan mengunduh hasil dataset ke format CSV.

## 🛠️ Persyaratan Instalasi
Pastikan Anda memiliki Python 3.10+ terinstal. Instal library yang diperlukan dengan perintah:

```bash
pip install -r requirements.txt