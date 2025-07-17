import streamlit as st
import pandas as pd
import pickle
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# --- FUNGSI PREPROCESSING (Sama seperti di Colab) ---
# Inisialisasi Stemmer & Stopword Remover
factory_stemmer = StemmerFactory()
stemmer = factory_stemmer.create_stemmer()

factory_stopword = StopWordRemoverFactory()
stopword_remover = factory_stopword.create_stop_word_remover()

# Fungsi untuk membersihkan dan memproses teks
def preprocess_text(text):
    # Case folding
    text = text.lower()
    # Menghapus karakter non-alfanumerik
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Stopword removal
    text = stopword_remover.remove(text)
    # Stemming
    text = stemmer.stem(text)
    return text

# --- LOAD MODEL & VECTORIZER ---
# Menggunakan cache untuk mempercepat loading model
@st.cache_resource
def load_model_and_vectorizer():
    with open('model_svm.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# --- ANTARMUKA STREAMLIT ---
# GANTI TEKS DI BAWAH INI SESUAI PROGRAM ANDA
st.set_page_config(page_title="Analisis Sentimen Aplikasi investasi di Playstore", page_icon="⚙️", layout="wide")

# GANTI TEKS DI BAWAH INI SESUAI PROGRAM ANDA
st.title("⚙️ Analisis Sentimen Aplikasi investasi di Playstore")
st.markdown("Aplikasi ini menggunakan model **Support Vector Machine (SVM)** untuk menganalisis sentimen dari sebuah teks.")

st.write("") # Spasi

# GANTI CONTOH TEKS DI BAWAH INI
user_input = st.text_area("Masukkan teks di sini:", "aplikasi yang dipakai sering down", height=150)

# Tombol untuk prediksi
if st.button("Analisis Sentimen", use_container_width=True, type="primary"):
    if user_input:
        # 1. Preprocess teks input
        preprocessed_input = preprocess_text(user_input)

        # 2. Vektorisasi teks
        vectorized_input = vectorizer.transform([preprocessed_input])

        # 3. Prediksi menggunakan model
        prediction = model.predict(vectorized_input)

        # Konversi label numerik ke teks (pastikan urutannya benar!)
        sentiment_map = {0: 'Negatif 👎', 1: 'Netral 😐', 2: 'Positif 👍'}
        result = sentiment_map.get(prediction[0], 'Tidak diketahui')

        st.write("")
        st.subheader("Hasil Analisis:")
        if result == 'Positif 👍':
            st.success(f"Sentimen: **{result}**")
        elif result == 'Negatif 👎':
            st.error(f"Sentimen: **{result}**")
        else:
            st.warning(f"Sentimen: **{result}**")

        # --- FITUR BARU YANG DITAMBAHKAN ---
        st.write("")
        with st.expander("Lihat Teks yang Diproses Model"):
            st.info(preprocessed_input)
        # ------------------------------------
            
    else:
        st.warning("Mohon masukkan teks terlebih dahulu.")

# GANTI NAMA FILE CSV ANDA DI SINI
with st.expander("Lihat Detail Data Latih"):
    try:
        df = pd.read_csv('Data_ulasan_Ranking_1.csv')
        st.write(df.head())
        sentimen_counts = df['sentimen'].value_counts()
        st.bar_chart(sentimen_counts)
    except FileNotFoundError:
        st.error("File dataset tidak ditemukan.")