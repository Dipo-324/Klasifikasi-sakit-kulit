import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="DermoScan AI", # Nama yang lebih catchy
    page_icon="🩺",
    layout="wide", # Menggunakan layout wide agar lebih luas
    initial_sidebar_state="expanded"
)

# --- CSS KUSTOM (Opsional: Untuk mempercantik tampilan) ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI LOAD MODEL & KELAS ---
@st.cache_resource
def load_model_and_classes():
    try:
        model = tf.keras.models.load_model('skin_disease_model.keras')
    except Exception as e:
        st.error(f"⚠️ Error memuat model: {e}")
        return None, None

    try:
        with open('class_names.json', 'r') as f:
            class_names = json.load(f)
    except Exception as e:
        st.error(f"⚠️ Error memuat class names: {e}")
        return None, None
    
    return model, class_names

# --- FUNGSI PREPROCESSING ---
def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- SIDEBAR (Instruksi & Info) ---
def sidebar_info():
    st.sidebar.title("ℹ️ Tentang Aplikasi")
    st.sidebar.info(
        """
        Aplikasi ini menggunakan **Deep Learning** untuk mendeteksi jenis penyakit kulit dari gambar yang diunggah.
        
        **Cara Penggunaan:**
        1. Upload foto bagian kulit yang ingin diperiksa.
        2. Klik tombol **'Analisis Kulit'**.
        3. Lihat hasil prediksi AI.
        """
    )
    st.sidebar.warning(
        """
        **DISCLAIMER PENTING:**
        Hasil ini adalah prediksi kecerdasan buatan (AI) dan **bukan** diagnosa medis mutlak.
        Segera konsultasikan dengan dokter kulit untuk pemeriksaan lebih lanjut.
        """
    )

# --- LOGIKA UTAMA ---
def main():
    sidebar_info()

    # Header Utama
    st.title("🩺 DermoScan: Deteksi Dini Penyakit Kulit")
    st.write("Unggah citra klinis kulit untuk mendapatkan analisis cepat berbasis AI.")
    st.divider()

    model, class_names = load_model_and_classes()
    if not model or not class_names:
        return

    # Layout Kolom Utama
    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.subheader("1. Unggah Gambar")
        uploaded_file = st.file_uploader("Format: JPG, PNG", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            # Tampilkan gambar dengan border rounded agar rapi
            st.image(image, caption='Preview Gambar', use_column_width=True)

    with col2:
        st.subheader("2. Hasil Analisis")
        
        # Placeholder agar UI tidak kosong saat belum ada prediksi
        if uploaded_file is None:
            st.info("Silakan unggah gambar di kolom sebelah kiri untuk melihat hasil prediksi di sini.")
        
        else:
            # Tombol Prediksi (Full Width karena CSS di atas)
            if st.button('🔍 Analisis Kulit'):
                with st.spinner('Sedang memproses citra...'):
                    # Preprocess & Prediksi
                    processed_image = preprocess_image(image)
                    predictions = model.predict(processed_image)
                    score = tf.nn.softmax(predictions[0])
                    
                    # Hasil Utama
                    top_idx = np.argmax(score)
                    predicted_class = class_names[top_idx]
                    confidence = 100 * np.max(score)

                    # Tampilan Hasil (Card Style)
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    
                    # # Logika Warna Alert
                    # if confidence > 80:
                    #     st.success(f"### Terdeteksi: {predicted_class}")
                    #     st.caption("AI memiliki keyakinan tinggi terhadap hasil ini.")
                    # elif confidence > 50:
                    #     st.warning(f"### Kemungkinan: {predicted_class}")
                    #     st.caption("AI cukup yakin, namun disarankan verifikasi ulang.")
                    # else:
                    #     st.error(f"### Tidak Yakin: {predicted_class}")
                    #     st.caption("Tingkat keyakinan rendah. Citra mungkin kurang jelas.")

                    # # Metric Display
                    # st.metric(label="Tingkat Keyakinan (Confidence)", value=f"{confidence:.2f}%")
                    # st.markdown('</div>', unsafe_allow_html=True)

                    # st.divider()

                    # Detail Probabilitas (Top 3)
                    st.write("#### Detail Probabilitas:")
                    top_3_indices = np.argsort(predictions[0])[-3:][::-1]
                    
                    for i in top_3_indices:
                        cls_name = class_names[i]
                        prob = float(predictions[0][i]) # Raw probability for progress bar
                        
                        # Layout baris untuk nama kelas dan progress bar
                        c_name, c_bar, c_val = st.columns([2, 3, 1])
                        with c_name:
                            st.write(f"**{cls_name}**")
                        with c_bar:
                            st.progress(prob) # Streamlit progress terima 0.0 - 1.0
                        with c_val:
                            st.write(f"{prob*100:.1f}%")

if __name__ == '__main__':
    main()