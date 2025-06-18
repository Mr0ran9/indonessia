import streamlit as st
st.set_page_config(page_title="Prediksi Jurusan", layout="centered")


import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# === Load data dan model ===
hasil_path = 'hasil_prediksi.csv'

try:
    with open('scalar_jurusan.pkl', 'rb') as f:
        loaded_data = pickle.load(f)

    if isinstance(loaded_data, tuple):
        df = loaded_data[0]
        scaler = loaded_data[1]
    else:
        df = loaded_data
        scaler = StandardScaler()
        X = df[['IPA', 'IPS', 'PKN', 'MTK']]
        scaler.fit(X)

    df['jurusan'] = ((df['IPA'] + df['MTK']) > (df['IPS'] + df['PKN'])).astype(int)
    X = df[['IPA', 'IPS', 'PKN', 'MTK']]
    y = df['jurusan']

    model = Sequential([
        Dense(8, input_dim=4, activation='relu'),
        Dense(4, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y, epochs=50, verbose=0)

except Exception as e:
    st.error(f"Error loading model: {e}")
    df = pd.DataFrame(columns=['IPA', 'IPS', 'PKN', 'MTK'])
    scaler = StandardScaler()
    model = Sequential([
        Dense(8, input_dim=4, activation='relu'),
        Dense(4, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# === Fungsi prediksi ===
def prediksi_dan_simpan(nisn, ipa, ips, pkn, mtk):
    try:
        # Pastikan NISN selalu string
        nisn = str(nisn).strip()

        input_data = pd.DataFrame([[ipa, ips, pkn, mtk]], columns=['IPA', 'IPS', 'PKN', 'MTK'])
        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0][0]
        jurusan = 'IPA' if pred >= 0.5 else 'IPS'

        new_entry = pd.DataFrame([{
            'NISN': nisn,
            'IPA': ipa,
            'IPS': ips,
            'PKN': pkn,
            'MTK': mtk,
            'Prediksi': jurusan
        }])

        if os.path.exists(hasil_path):
            hasil_df = pd.read_csv(hasil_path, dtype={'NISN': str})  # pastikan NISN dibaca sebagai string
            if nisn in hasil_df['NISN'].values:
                return "NISN_SUDAH_ADA"
            hasil_df = pd.concat([hasil_df, new_entry], ignore_index=True)
        else:
            hasil_df = new_entry

        hasil_df.to_csv(hasil_path, index=False)
        return jurusan
    except Exception as e:
        return f"ERROR: {e}"

# === UI Streamlit ===
st.title("📚 Prediksi Jurusan Siswa SMA")

with st.form("form_prediksi"):
    nisn = st.text_input("Masukkan NISN:")
    ipa = st.number_input("Nilai IPA", 0.0, 100.0, step=0.1)
    ips = st.number_input("Nilai IPS", 0.0, 100.0, step=0.1)
    pkn = st.number_input("Nilai PKN", 0.0, 100.0, step=0.1)
    mtk = st.number_input("Nilai Matematika", 0.0, 100.0, step=0.1)

    submitted = st.form_submit_button("🔮 Prediksi Jurusan")

if submitted:
    if nisn.strip() == "":
        st.warning("⚠️ NISN tidak boleh kosong.")
    else:
        hasil = prediksi_dan_simpan(nisn.strip(), ipa, ips, pkn, mtk)
        if hasil == "NISN_SUDAH_ADA":
            st.error("❌ NISN ini sudah pernah dipakai untuk prediksi.")
        elif hasil.startswith("ERROR"):
            st.error("Terjadi kesalahan saat memproses prediksi.")
            st.text(hasil)
        else:
            st.success(f"✅ Prediksi jurusan untuk NISN {nisn}: **{hasil}**")
            # Scroll otomatis ke bawah (workaround HTML/JS)
st.markdown("""
    <script>
        const predictionResult = document.querySelector('section.main');
        if (predictionResult) {
            setTimeout(() => {
                predictionResult.scrollTo({ top: predictionResult.scrollHeight, behavior: 'smooth' });
            }, 500);
        }
    </script>
""", unsafe_allow_html=True)

# === Tampilkan Riwayat ===
st.subheader("📄 Riwayat Prediksi")

if os.path.exists(hasil_path):
    hasil_df = pd.read_csv(hasil_path, dtype={'NISN': str})
    hasil_df = hasil_df.sort_values(by='NISN', ascending=False)  # Urutkan dari NISN terbesar ke terkecil
    st.dataframe(hasil_df, use_container_width=True)
else:
    st.info("Belum ada riwayat prediksi.")

