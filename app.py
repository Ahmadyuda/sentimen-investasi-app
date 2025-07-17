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
user_input = st.text_area("Masukkan teks di sini:", "aplikasi yang dipakai lancar", height=150)

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
        df = pd.read_csv('Data ulasan Ranking 1.csv')
        st.write(df.head())
        sentimen_counts = df['sentimen'].value_counts()
        st.bar_chart(sentimen_counts)
    except FileNotFoundError:
        st.error("File dataset tidak ditemukan.")

# Daftar aplikasi yang dianalisis
APPS_INFO = {
    'Ajaib': 'ajaib.co.id',
    'Bibit': 'com.bibit.bibitid',
    'Bareksa': 'com.bareksa.app',
    'IPOT (IndoPremier)': 'com.indopremier.ipot',
    'Stockbit': 'com.stockbit.android',
    'Pluang': 'com.EmasDigi',
    'MOST (Mandiri Sekuritas)': 'com.mandirisekuritas.most',
    'BIONS (BNI Sekuritas)': 'id.zaisan.android',
    'RHB Tradesmart ID': 'com.rhbsyariah.tradesmart',
    'POEMS ID (Phillip Sekuritas)': 'com.phillip.prima',
    'Mirae HOTS Mobile': 'id.co.miraeassetdaewoo',
    'Trima (Trimegah Sekuritas)': 'com.trimegah.trima',
    'MotionTrade (MNC Sekuritas)': 'com.mncsecurities.mnctrade',
    'CGS-CIMB iTrade': 'id.co.cimbniaga.mobile.android',
    'Sinarmas Sekuritas': 'com.simas.siminvest',
    'IDX Mobile': 'id.co.idx.idxmobile',
    'Nanovest': 'com.nanovest.prod',
    'Hero Investment': 'kr.co.daou.kiwoomherosg',
    'InvestASIK (Danareksa)': 'com.danareksa.investasik',
    'BEST Mobile (BCA Sekuritas)': 'com.bcasekuritas.mybest'
}

# Letakkan di bagian utama app.py, misalnya di paling bawah

import pandas as pd # Pastikan pandas sudah di-import

st.write("---") # Garis pemisah
with st.expander("Lihat Daftar Lengkap Aplikasi yang Dianalisis"):
    # Ubah dictionary menjadi DataFrame pandas
    df_apps = pd.DataFrame(list(APPS_INFO.items()), columns=['Nama Aplikasi', 'ID Paket Google Play'])
    st.dataframe(df_apps, use_container_width=True, hide_index=True)

# Pastikan DataFrame 'new_df_copy' dan kolom yang dibutuhkan sudah ada
if 'new_df_copy' in locals() and all(col in new_df_copy.columns for col in ['sentimen', 'tokens_akhir']):

    print("✅ Memfilter data dengan sentimen 'Negatif'...")

    # Filter baris dengan sentimen 'Negatif' dan pilih kolom 'tokens_akhir' & 'sentimen'
    train_s0 = new_df_copy.loc[new_df_copy['sentimen'] == 'Negatif', ['tokens_akhir', 'sentimen']].copy()

    print("🎉 DataFrame 'train_s0' berhasil dibuat.")

else:
    print("❌ Gagal membuat 'train_s0'. Pastikan langkah sebelumnya sudah dijalankan.")

# ===================================================================
#           MEMBUAT WORDCLOUD DARI DATA BERSIH (FIXED)
# ===================================================================

# Pastikan DataFrame train_s0 dan library yang dibutuhkan sudah ada
if 'train_s0' in locals():

    # Instalasi library jika belum ada
    !pip install wordcloud -q

    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    print("✅ DataFrame 'train_s0' ditemukan. Membuat WordCloud...")

    # ================== BAGIAN YANG DIPERBAIKI ==================
    # 1. Ubah setiap list di 'tokens_akhir' menjadi kalimat (string)
    #    Ini menciptakan kolom baru yang berisi kalimat bersih.
    train_s0['text_final'] = train_s0['tokens_akhir'].apply(lambda tokens: ' '.join(tokens))

    # 2. Gabungkan semua kalimat dari kolom 'text_final' yang baru
    all_text_s0 = ' '.join(train_s0['text_final'])
    # ==========================================================

    # 3. Buat objek WordCloud dan hasilkan gambarnya
    #    Karena all_text_s0 sudah bersih, wordcloud akan lebih informatif
    wordcloud = WordCloud(colormap='Reds', width=1000, height=500, mode='RGBA', background_color='white').generate(all_text_s0)

    # 4. Tampilkan WordCloud menggunakan matplotlib
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("WordCloud Sentimen Negatif (dari Teks Bersih)")
    plt.show()

else:
    print("❌ DataFrame 'train_s0' tidak ditemukan. Jalankan sel sebelumnya terlebih dahulu.")

# Pastikan DataFrame 'new_df_copy' dan kolom yang dibutuhkan sudah ada
if 'new_df_copy' in locals() and all(col in new_df_copy.columns for col in ['sentimen', 'tokens_akhir']):

    print("✅ Memfilter data dengan sentimen 'Netral'...")

    # Filter baris dengan sentimen 'Netral' dan pilih kolom 'tokens_akhir' & 'sentimen'
    train_s1 = new_df_copy.loc[new_df_copy['sentimen'] == 'Netral', ['tokens_akhir', 'sentimen']].copy()

    print("🎉 DataFrame 'train_s1' berhasil dibuat.")

else:
    print("❌ Gagal membuat 'train_s1'. Pastikan langkah sebelumnya sudah dijalankan.")

# ===================================================================
#           MEMBUAT WORDCLOUD DARI DATA BERSIH (FIXED)
# ===================================================================

# Pastikan DataFrame train_s0 dan library yang dibutuhkan sudah ada
if 'train_s1' in locals():

    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    print("✅ DataFrame 'train_s1' ditemukan. Membuat WordCloud...")

    # ================== BAGIAN YANG DIPERBAIKI ==================
    # 1. Ubah setiap list di 'tokens_akhir' menjadi kalimat (string)
    #    Ini menciptakan kolom baru yang berisi kalimat bersih.
    train_s1['text_final'] = train_s1['tokens_akhir'].apply(lambda tokens: ' '.join(tokens))

    # 2. Gabungkan semua kalimat dari kolom 'text_final' yang baru
    all_text_s1 = ' '.join(train_s1['text_final'])
    # ==========================================================

    # 3. Buat objek WordCloud dan hasilkan gambarnya
    #    Karena all_text_s0 sudah bersih, wordcloud akan lebih informatif
    wordcloud = WordCloud(colormap='Blues', width=1000, height=500, mode='RGBA', background_color='white').generate(all_text_s0)

    # 4. Tampilkan WordCloud menggunakan matplotlib
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("WordCloud Sentimen Netral (dari Teks Bersih)")
    plt.show()

else:
    print("❌ DataFrame 'train_s1' tidak ditemukan. Jalankan sel sebelumnya terlebih dahulu.")

# Pastikan DataFrame 'new_df_copy' dan kolom yang dibutuhkan sudah ada
if 'new_df_copy' in locals() and all(col in new_df_copy.columns for col in ['sentimen', 'tokens_akhir']):

    print("✅ Memfilter data dengan sentimen 'Positif'...")

    # Filter baris dengan sentimen 'Positif' dan pilih kolom 'tokens_akhir' & 'sentimen'
    train_s2 = new_df_copy.loc[new_df_copy['sentimen'] == 'Positif', ['tokens_akhir', 'sentimen']].copy()

    print("🎉 DataFrame 'train_s2' berhasil dibuat.")

else:
    print("❌ Gagal membuat 'train_s2'. Pastikan langkah sebelumnya sudah dijalankan.")

# ===================================================================
#           MEMBUAT WORDCLOUD DARI DATA BERSIH (FIXED)
# ===================================================================

# Pastikan DataFrame train_s0 dan library yang dibutuhkan sudah ada
if 'train_s2' in locals():

    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    print("✅ DataFrame 'train_s2' ditemukan. Membuat WordCloud...")

    # ================== BAGIAN YANG DIPERBAIKI ==================
    # 1. Ubah setiap list di 'tokens_akhir' menjadi kalimat (string)
    #    Ini menciptakan kolom baru yang berisi kalimat bersih.
    train_s2['text_final'] = train_s2['tokens_akhir'].apply(lambda tokens: ' '.join(tokens))

    # 2. Gabungkan semua kalimat dari kolom 'text_final' yang baru
    all_text_s2 = ' '.join(train_s2['text_final'])
    # ==========================================================

    # 3. Buat objek WordCloud dan hasilkan gambarnya
    #    Karena all_text_s0 sudah bersih, wordcloud akan lebih informatif
    wordcloud = WordCloud(colormap='Greens', width=1000, height=500, mode='RGBA', background_color='white').generate(all_text_s0)

    # 4. Tampilkan WordCloud menggunakan matplotlib
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("WordCloud Sentimen Positif (dari Teks Bersih)")
    plt.show()

else:
    print("❌ DataFrame 'train_s2' tidak ditemukan. Jalankan sel sebelumnya terlebih dahulu.")
