import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="Prediksi Risiko Kredit",
    layout="wide"
)

try:
    model = joblib.load('credit_risk_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')
except FileNotFoundError:
    st.error("File model/scaler/kolom tidak ditemukan. Pastikan file .pkl ada di direktori yang sama.")
    st.stop()


map_account_balance = {
    "Tidak punya rekening giro": 1,
    "Saldo negatif (< 0 DM)": 2,
    "Saldo 0 - 200 DM": 3,
    "Saldo > 200 DM": 4
}
map_payment_status = {
    "Semua kredit lunas sempurna": 4,
    "Riwayat pembayaran di bank ini lancar": 3,
    "Riwayat pembayaran lancar (di tempat lain)": 2,
    "Ada kredit kritis / di bank lain": 1,
    "Pernah macet/tertunda": 0
}
map_purpose = {
    'Mobil Baru': 0, 'Mobil Bekas': 1, 'Perabotan/Interior': 2, 'Radio/TV': 3,
    'Elektronik Rumah Tangga': 4, 'Perbaikan/Renovasi': 5, 'Pendidikan': 6,
    'Pelatihan': 8, 'Bisnis': 9, 'Lainnya': 10
}
map_savings = {
    "Tidak punya tabungan": 1,
    "< 100 DM": 2,
    "100 - 500 DM": 3,
    "500 - 1000 DM": 4,
    "> 1000 DM": 5
}
map_employment = {
    "Menganggur": 1,
    "< 1 tahun": 2,
    "1 - 4 tahun": 3,
    "4 - 7 tahun": 4,
    "> 7 tahun": 5
}
map_sex_marital = {
    "Pria : Lajang": 3,
    "Pria : Menikah/Duda": 4,
    "Pria : Cerai/Pisah": 1,
    "Wanita : Cerai/Pisah/Menikah": 2
}
map_guarantors = {
    "Tidak Ada": 1,
    "Rekan Pemohon (Co-applicant)": 2,
    "Penjamin (Guarantor)": 3
}
map_asset = {
    "Properti (Rumah/Tanah)": 4,
    "Asuransi Jiwa / Kontrak Pembangunan": 3,
    "Mobil atau Lainnya": 2,
    "Tidak Diketahui / Tidak Punya Aset": 1
}
map_concurrent_credits = {
    "Tidak Ada": 3,
    "Di Bank Lain": 2,
    "Di Toko Lain": 1
}
map_apartment = {
    "Sewa": 1,
    "Milik Sendiri": 2,
    "Gratis (Tinggal dengan Keluarga)": 3
}
map_occupation = {
    "Manajer/Wiraswasta/Sangat Terampil": 4,
    "Karyawan Terampil / PNS": 3,
    "Tidak Terampil & Menetap": 2,
    "Tidak Terampil & Tidak Menetap": 1
}
map_telephone = { "Tidak Ada": 1, "Ada (Terdaftar atas nama nasabah)": 2 }
map_foreign_worker = { "Ya": 1, "Tidak": 2 }

st.sidebar.title("Masukkan Data Nasabah:")

def user_input_features():
    
    account_balance_label = st.sidebar.selectbox('Saldo Rekening', list(map_account_balance.keys()))
    payment_status_label = st.sidebar.selectbox('Riwayat Pembayaran Sebelumnya', list(map_payment_status.keys()))
    purpose_label = st.sidebar.selectbox('Tujuan Kredit', list(map_purpose.keys()))
    savings_label = st.sidebar.selectbox('Tabungan / Saham', list(map_savings.keys()))
    employment_label = st.sidebar.selectbox('Lama Bekerja Saat Ini', list(map_employment.keys()))
    sex_marital_label = st.sidebar.selectbox('Status Pernikahan & Jenis Kelamin', list(map_sex_marital.keys()))
    guarantors_label = st.sidebar.selectbox('Penjamin Lainnya', list(map_guarantors.keys()))
    asset_label = st.sidebar.selectbox('Aset Paling Berharga', list(map_asset.keys()))
    concurrent_credits_label = st.sidebar.selectbox('Kredit Lain yang Sedang Berjalan', list(map_concurrent_credits.keys()))
    apartment_label = st.sidebar.selectbox('Tipe Tempat Tinggal', list(map_apartment.keys()))
    occupation_label = st.sidebar.selectbox('Pekerjaan', list(map_occupation.keys()))
    telephone_label = st.sidebar.selectbox('Telepon', list(map_telephone.keys()))
    foreign_worker_label = st.sidebar.selectbox('Pekerja Asing', list(map_foreign_worker.keys()))
    
    duration_monthly = st.sidebar.slider('Durasi Kredit (Bulan)', 4, 72, 18)
    credit_amount = st.sidebar.number_input('Jumlah Kredit (DM)', min_value=250, max_value=20000, value=1500, step=50)
    age = st.sidebar.slider('Umur Nasabah (Tahun)', 19, 75, 35)
    instalment_percent = st.sidebar.slider('Persentase Cicilan dari Gaji (%)', 1, 4, 2, help="Persentase cicilan dari pendapatan yang bisa dibuang.")
    duration_address = st.sidebar.slider('Lama Tinggal di Alamat Saat Ini (Tahun)', 1, 4, 2)
    no_credits_at_bank = st.sidebar.slider('Jumlah Kredit di Bank Ini', 1, 4, 1)
    no_dependents = st.sidebar.slider('Jumlah Tanggungan', 1, 2, 1)
    
    data = {
        'Account_Balance': map_account_balance[account_balance_label],
        'Duration_of_Credit_monthly': duration_monthly,
        'Payment_Status_of_Previous_Credit': map_payment_status[payment_status_label],
        'Purpose': map_purpose[purpose_label],
        'Credit_Amount': credit_amount,
        'Value_Savings_Stocks': map_savings[savings_label],
        'Length_of_current_employment': map_employment[employment_label],
        'Instalment_per_cent': instalment_percent,
        'Sex_Marital_Status': map_sex_marital[sex_marital_label],
        'Guarantors': map_guarantors[guarantors_label],
        'Duration_in_Current_address': duration_address,
        'Most_valuable_available_asset': map_asset[asset_label],
        'Age_years': age,
        'Concurrent_Credits': map_concurrent_credits[concurrent_credits_label],
        'Type_of_apartment': map_apartment[apartment_label],
        'No_of_Credits_at_this_Bank': no_credits_at_bank,
        'Occupation': map_occupation[occupation_label],
        'No_of_dependents': no_dependents,
        'Telephone': map_telephone[telephone_label],
        'Foreign_Worker': map_foreign_worker[foreign_worker_label]
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()


st.title("Aplikasi Prediksi Risiko Kredit Nasabah")
st.markdown("""
Aplikasi ini menggunakan model *Machine Learning* (Random Forest yang telah dioptimalkan) untuk memprediksi apakah seorang nasabah layak mendapatkan kredit **(Good Risk)** atau berisiko gagal bayar **(Bad Risk)**.
Gunakan *sidebar* di sebelah kiri untuk memasukkan data nasabah baru dan lihat hasilnya.
""")

with st.expander("Lihat Data Numerik yang Dikirim ke Model"):
    st.dataframe(input_df.style.highlight_max(axis=1))

if st.sidebar.button('Prediksi Kelayakan'):
    input_df_reordered = input_df[model_columns]
    
    scaled_input = scaler.transform(input_df_reordered)
    
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    st.markdown("---")
    st.subheader('Hasil Prediksi:')
    
    col1, col2 = st.columns(2)
    with col1:
        if prediction[0] == 1:
            st.success('**Status: Layak Diberi Kredit (Good Risk)**')
        else:
            st.error('**Status: Berisiko Gagal Bayar (Bad Risk)**')
            st.warning("Disarankan untuk melakukan evaluasi lebih lanjut terhadap nasabah ini.")
    
    with col2:
        if prediction[0] == 1:
            st.metric(label="Tingkat Kepercayaan (Good Risk)", value=f"{prediction_proba[0][1]*100:.2f}%")
        else:
            st.metric(label="Tingkat Kepercayaan (Bad Risk)", value=f"{prediction_proba[0][0]*100:.2f}%")


st.markdown("---")

st.header("Analisis dan Performa Model")
st.write("Bagian ini menunjukkan analisis data dan perbandingan model yang telah dilakukan untuk membangun sistem prediksi ini.")

tab1, tab2 = st.tabs(["Faktor Risiko", "Performa Model"])

with tab1:
    st.subheader("Analisis Risiko Berdasarkan Faktor Kunci")
    st.markdown("""
    Analisis data historis menunjukkan bahwa beberapa faktor lebih signifikan dalam menentukan risiko kredit. 
    **Riwayat Pembayaran Sebelumnya** adalah salah satu prediktor terkuat. Nasabah dengan riwayat pembayaran yang sempurna memiliki tingkat kelayakan yang jauh lebih tinggi.
    """)
    try:
        st.image('asets/risk_by_history.png', caption='Persentase Kelayakan Kredit Berdasarkan Riwayat Pembayaran')
    except FileNotFoundError:
        st.warning("File gambar 'risk_by_history.png' tidak ditemukan.")

with tab2:
    st.subheader("Perbandingan Kinerja Model (Default vs. Tuned)")
    st.markdown("""
    Beberapa model Machine Learning diuji untuk menemukan yang terbaik. Proses **Hyperparameter Tuning** dilakukan untuk mengoptimalkan model-model yang paling potensial. 
    Grafik di bawah menunjukkan peningkatan skor **ROC AUC** setelah proses tuning.
    """)
    try:
        st.image('asets/tuning_comparison.png', caption='Peningkatan Performa Model Setelah Tuning')
    except FileNotFoundError:
        st.warning("File gambar 'tuning_comparison.png' tidak ditemukan.")
    
    st.markdown("#### Leaderboard Akhir")
    st.write("Tabel berikut menunjukkan performa akhir dari semua model yang diuji pada data tes. **Random Forest_Tuned** terpilih sebagai model final karena memiliki skor ROC AUC tertinggi.")
    try:
        leaderboard_df = pd.read_csv('leaderboard_final.csv')
        st.dataframe(leaderboard_df.style.highlight_max(subset=['ROC_AUC'], color='lightgreen'))
    except FileNotFoundError:
        st.warning("File 'leaderboard_final.csv' tidak ditemukan.")