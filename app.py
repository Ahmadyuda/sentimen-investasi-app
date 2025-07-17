import streamlit as st
import pandas as pd
import pickle
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from google_play_scraper import app

# --- KONSTANTA & INISIALISASI AWAL ---

# Daftar aplikasi yang akan dianalisis dan di-scrape
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

# Inisialisasi Sastrawi (Stemmer & Stopword Remover)
factory_stemmer = StemmerFactory()
stemmer = factory_stemmer.create_stemmer()
factory_stopword = StopWordRemoverFactory()
stopword_remover = factory_stopword.create_stop_word_remover()


# --- FUNGSI-FUNGSI ---

def preprocess_text(text):
    """Fungsi untuk membersihkan dan memproses teks input."""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    return text

@st.cache_resource
def load_model_and_vectorizer():
    """Memuat model dan vectorizer dari file .pkl dengan cache."""
    with open('model_svm.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# --- MEMUAT MODEL ---
model, vectorizer = load_model_and_vectorizer()


# --- KONFIGURASI HALAMAN UTAMA ---
st.set_page_config(page_title="Analisis Aplikasi Investasi", page_icon="📈", layout="wide")


# --- BAGIAN 1: ANALISIS SENTIMEN ---
st.title("📈 Analisis Sentimen Aplikasi Investasi di Play Store")
st.markdown("Aplikasi ini menggunakan model **Support Vector Machine (SVM)** untuk menganalisis sentimen dari sebuah teks ulasan.")
user_input = st.text_area("Masukkan teks ulasan di sini:", "aplikasi yang dipakai lancar dan mudah digunakan", height=150)

if st.button("Analisis Sentimen", use_container_width=True, type="primary"):
    if user_input:
        # 1. Pisahkan input menjadi beberapa baris/komentar dan abaikan baris kosong.
        comments = [line.strip() for line in user_input.split('\n') if line.strip()]

        if comments:
            results = []
            # 2. Loop melalui setiap komentar untuk dianalisis
            for comment in comments:
                preprocessed_input = preprocess_text(comment)
                vectorized_input = vectorizer.transform([preprocessed_input])
                prediction = model.predict(vectorized_input)
                
                sentiment_map = {0: 'Negatif 👎', 1: 'Netral 😐', 2: 'Positif 👍'}
                result_text = sentiment_map.get(prediction[0], 'Tidak diketahui')
                
                # Simpan hasil untuk setiap komentar
                results.append({'Komentar': comment, 'Prediksi Sentimen': result_text})

            # 3. Buat DataFrame dari hasil dan hitung ringkasan sentimen
            df_results = pd.DataFrame(results)
            sentiment_counts = df_results['Prediksi Sentimen'].value_counts()
            
            st.subheader("📊 Ringkasan Hasil Analisis")
            
            # Format ringkasan untuk ditampilkan dengan rapi
            summary_list = [f"{count} {label.split(' ')[0]}" for label, count in sentiment_counts.items()]
            summary_text = ", ".join(summary_list)
            st.success(f"**Hasil: {summary_text}**")
            
            # 4. Tampilkan tabel detail analisis per komentar
            st.subheader("Detail Analisis per Komentar")
            st.dataframe(df_results, use_container_width=True, hide_index=True)

        else:
            st.warning("Mohon masukkan setidaknya satu komentar yang valid.")
    else:
        st.warning("Area input masih kosong. Mohon masukkan teks terlebih dahulu.")


# Garis pemisah antar bagian
st.write("---")


# --- BAGIAN 2: PERINGKAT APLIKASI (LIVE SCRAPING) ---
st.header("🏆 Peringkat Aplikasi Investasi Berdasarkan Data Play Store")
all_app_summary_data = []
total_apps = len(APPS_INFO)

progress_bar = st.progress(0, text="Memulai proses pengambilan data...")
status_text = st.empty()

# Loop untuk scraping data
for i, (nama_app, id_app) in enumerate(APPS_INFO.items()):
    status_text.text(f"Mengambil data untuk: {nama_app} ({i+1}/{total_apps})...")
    try:
        result = app(id_app, lang='id', country='id')
        all_app_summary_data.append(result)
    except Exception as e:
        st.warning(f"Gagal mengambil data untuk {nama_app}.")
    progress_bar.progress((i + 1) / total_apps)

status_text.success("Pengambilan data peringkat selesai!")

# Proses dan tampilkan hasil scraping jika berhasil
if all_app_summary_data:
    df_summary = pd.DataFrame(all_app_summary_data)
    df_summary.dropna(subset=['score'], inplace=True)
    df_sorted_summary = df_summary.sort_values(by=['score', 'ratings'], ascending=[False, False]).reset_index(drop=True)
    df_sorted_summary['Ranking'] = df_sorted_summary.index + 1

    st.dataframe(
        df_sorted_summary[['Ranking', 'title', 'score', 'ratings', 'installs', 'developer']],
        hide_index=True,
        use_container_width=True
    )
else:
    st.error("Tidak ada data aplikasi yang berhasil diambil. Tidak dapat menampilkan peringkat.")


# Garis pemisah antar bagian
st.write("---")

# --- BAGIAN 3: DETAIL DATA LATIH (DARI FILE CSV) ---
with st.expander("Lihat Detail Data Latih yang Digunakan Model"):
    try:
        df_latih = pd.read_csv('Data ulasan Ranking 1.csv')
        st.write("Contoh Data:", df_latih.head(1000))
        
        st.subheader("Distribusi Sentimen pada Data Latih")
        sentimen_counts = df_latih['sentimen'].value_counts()
        st.bar_chart(sentimen_counts)
    except FileNotFoundError:
        st.error("File 'Data ulasan Ranking 1.csv' tidak ditemukan di repository. Detail data latih tidak dapat ditampilkan.")
    except KeyError:
        st.error("Kolom 'sentimen' tidak ditemukan di file CSV. Tidak dapat menampilkan distribusi sentimen.")
