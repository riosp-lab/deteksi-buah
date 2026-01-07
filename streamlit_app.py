import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import io
import time
import zipfile
import tempfile
import shutil
import plotly.graph_objects as go
import gdown

# --- 1. IMPORT DATA ---
try:
    from nutrisi import CLASS_NAMES, NUTRISI_DATA
except ImportError:
    st.error("File 'nutrisi.py' tidak ditemukan!")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# 2. LOAD MODEL (Sesuai Struktur Colab Kamu)
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🍎 FruitScan AI",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)
APP_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_CACHE_ROOT = os.path.join(tempfile.gettempdir(), "fruit_scan_model")
# Folder awal hanya placeholder; setelah ekstrak ZIP, MODEL_DIR akan diarahkan ke folder
# yang benar (yang berisi saved_model.pb) lewat hasil scan/marker.
MODEL_DIR = os.path.join(MODEL_CACHE_ROOT, "model")
MODEL_DIR_MARKER = os.path.join(MODEL_CACHE_ROOT, ".model_dir_path")
GOOGLE_DRIVE_ID = "1Lli5EyHbikpE10LoaQ0s9HE7j7vSH0RG"

os.makedirs(MODEL_CACHE_ROOT, exist_ok=True)

def _find_saved_model_dir(search_root: str):
    for root, _dirs, files in os.walk(search_root):
        if "saved_model.pb" in files:
            return root
    return None

def _safe_extract_zip(zip_path: str, dest_dir: str):
    dest_dir_abs = os.path.abspath(dest_dir)
    with zipfile.ZipFile(zip_path, "r") as z:
        for info in z.infolist():
            member_name = info.filename.replace("\\", "/")
            if member_name.endswith("/"):
                continue

            target_path = os.path.abspath(os.path.join(dest_dir, member_name))
            if not target_path.startswith(dest_dir_abs + os.sep) and target_path != dest_dir_abs:
                raise RuntimeError("Unsafe ZIP path detected")

            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with z.open(info, "r") as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

def ensure_model_ready():
    global MODEL_DIR

    if os.path.exists(MODEL_DIR_MARKER):
        try:
            with open(MODEL_DIR_MARKER, "r", encoding="utf-8") as f:
                saved_dir = f.read().strip()
            if saved_dir and os.path.exists(os.path.join(saved_dir, "saved_model.pb")):
                MODEL_DIR = saved_dir
        except Exception:
            pass

    if os.path.exists(os.path.join(MODEL_DIR, "saved_model.pb")):
        return MODEL_DIR

    zip_path = os.path.join(MODEL_CACHE_ROOT, "model.zip")
    if os.path.exists(zip_path):
        try:
            os.remove(zip_path)
        except Exception:
            pass

    with st.spinner("Mengunduh model dari Google Drive..."):
        gdown.download(id=GOOGLE_DRIVE_ID, output=zip_path, quiet=False)

    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError("File yang terunduh bukan ZIP yang valid")

    with st.spinner("Mengekstrak model..."):
        _safe_extract_zip(zip_path, MODEL_CACHE_ROOT)

    found_dir = _find_saved_model_dir(MODEL_CACHE_ROOT)
    if found_dir is None:
        raise RuntimeError("Model berhasil diunduh tapi saved_model.pb tidak ditemukan")

    MODEL_DIR = found_dir
    try:
        with open(MODEL_DIR_MARKER, "w", encoding="utf-8") as f:
            f.write(MODEL_DIR)
    except Exception:
        pass

    return MODEL_DIR

@st.cache_resource
def load_trained_model():
    local_pb = os.path.exists(os.path.join(APP_DIR, "saved_model.pb"))
    if local_pb:
        try:
            # MEMUAT MODEL (Format SavedModel)
            return tf.saved_model.load(APP_DIR)
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            return None

    try:
        model_dir = ensure_model_ready()
        return tf.saved_model.load(model_dir)
    except Exception as e:
        st.error(f"Gagal menyiapkan model: {e}")
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# 3. FUNGSI PREDIKSI (SINKRONISASI TOTAL DENGAN COLAB)
# ═══════════════════════════════════════════════════════════════════════════════
def preprocess_image(image: Image.Image):
    # 1. Resize Wajib 64x64 (Sesuai Cell 4 di notebook-mu)
    img = image.convert("RGB").resize((64, 64)) 
    img_array = np.array(img).astype(np.float32)
    
    # 2. JANGAN RESCALE MANUAL, JANGAN PREPROCESS_INPUT MANUAL
    # Karena di notebook-mu, model sudah punya layer tersebut di dalamnya.
    
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def model_predict(model, img_array):
    # PENTING: Memanggil signature 'serving_default'
    infer = model.signatures["serving_default"]
    input_name = list(infer.structured_input_signature[1].keys())[0]
    
    # PENTING SEKALI: Tambahkan training=False agar Augmentasi (Flip/Rotate) MATI
    # Jika tidak, gambar pisang kamu akan diputar-putar acak oleh AI
    predictions = infer(**{input_name: tf.constant(img_array)})
    
    output_name = list(predictions.keys())[0]
    return predictions[output_name].numpy()

def get_display_name(class_name: str) -> str:
    tokens = class_name.split()
    if tokens and tokens[-1].isdigit():
        return " ".join(tokens[:-1])
    return class_name

def get_fruit_color(name: str):
    name_lower = name.lower()
    if any(x in name_lower for x in [
        'apple red', 'apple crimson', 'apple hit', 'apple rotten',
        'banana red', 'blackberrie not rippen', 'cherry', 'strawberry', 'tomato',
        'cabbage red', 'onion red'
    ]):
        color1, color2 = "#ef4444", "#b91c1c"
        shadow_color = "rgba(239, 68, 68, 0.4)"
    elif any(x in name_lower for x in [
        'apple granny', 'avocado', 'beans', 'cabbage white', 'apple green',
        'kiwi', 'lime', 'pear', 'cucumber', 'watermelon'
    ]):
        color1, color2 = "#10b981", "#059669"
        shadow_color = "rgba(16, 185, 129, 0.4)"
    elif any(x in name_lower for x in [
        'apple golden', 'apricot', 'banana', 'lemon', 'orange',
        'cantaloupe', 'papaya', 'mango', 'peach', 'corn'
    ]):
        color1, color2 = "#f59e0b", "#d97706"
        shadow_color = "rgba(245, 158, 11, 0.4)"
    elif any(x in name_lower for x in [
        'blueberry', 'blackberrie', 'beetroot',
        'grape', 'plum', 'eggplant'
    ]):
        color1, color2 = "#8b5cf6", "#7c3aed"
        shadow_color = "rgba(139, 92, 246, 0.4)"
    elif any(x in name_lower for x in ['apple pink lady', 'peach', 'pitaya']):
        color1, color2 = "#ec4899", "#be185d"
        shadow_color = "rgba(236, 72, 153, 0.4)"
    elif any(x in name_lower for x in ['potato', 'ginger', 'chestnut', 'coconut']):
        color1, color2 = "#a8a29e", "#78716c"
        shadow_color = "rgba(168, 162, 158, 0.4)"
    else:
        if 'apple' in name_lower:
            color1, color2 = "#ef4444", "#b91c1c"
            shadow_color = "rgba(239, 68, 68, 0.4)"
        else:
            color1, color2 = "#10b981", "#059669"
            shadow_color = "rgba(16, 185, 129, 0.4)"
    return color1, color2, shadow_color

def create_nutrition_chart(nutrisi: dict):
    def _parse_float(value) -> float:
        if value is None:
            return 0.0
        s = str(value).strip().lower()
        s = s.replace(",", ".")
        allowed = set("0123456789.")
        filtered = "".join(ch for ch in s if ch in allowed)
        if filtered in ("", "."):
            return 0.0
        try:
            return float(filtered)
        except Exception:
            return 0.0

    kalori = _parse_float(nutrisi.get("Kalori (100g)", "0 kcal"))
    serat = _parse_float(nutrisi.get("Serat", "0 g"))
    fig = go.Figure(data=[go.Pie(
        labels=['Kalori', 'Serat', 'Lainnya'],
        values=[kalori, serat * 10, max(0, 100 - kalori - serat * 10)],
        hole=0.6,
        marker=dict(colors=['#667eea', '#10b981', '#e2e8f0']),
        textinfo='label+percent',
        textfont=dict(size=12, family='Inter'),
        hovertemplate='%{label}: %{value:.1f}<extra></extra>'
    )])
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        annotations=[dict(
            text=f'{kalori:.0f}<br>kcal',
            x=0.5, y=0.5,
            font=dict(size=20, family='Inter', color='#1e293b'),
            showarrow=False
        )]
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# 4. TAMPILAN UTAMA
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    if "input_mode" not in st.session_state:
        st.session_state.input_mode = "upload"
    if "uploaded_image_bytes" not in st.session_state:
        st.session_state.uploaded_image_bytes = None
    if "uploaded_image_sig" not in st.session_state:
        st.session_state.uploaded_image_sig = None
    if "last_prediction_sig" not in st.session_state:
        st.session_state.last_prediction_sig = None
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None

    st.markdown(
        "<div class='glass-card animate-fade-in'>"
        "<h1 class='hero-title'>🍎 FruitScan AI</h1>"
        "<p class='hero-subtitle'>"
        "Deteksi buah secara instan dengan AI dan dapatkan informasi nutrisi lengkap"
        "</p>"
        "</div>",
        unsafe_allow_html=True
    )

    col_input, col_result = st.columns([1, 1.3], gap="large")

    with col_input:
        st.markdown(
            "<div class='glass-card'>"
            "<div class='section-header'>Input Gambar</div>",
            unsafe_allow_html=True
        )

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button(
                "Upload File",
                use_container_width=True,
                type="primary" if st.session_state.input_mode == "upload" else "secondary",
            ):
                st.session_state.input_mode = "upload"
                st.rerun()

        with btn_col2:
            if st.button(
                "Kamera",
                use_container_width=True,
                type="primary" if st.session_state.input_mode == "kamera" else "secondary",
            ):
                st.session_state.input_mode = "kamera"
                st.rerun()

        uploaded_file = None
        if st.session_state.input_mode == "upload":
            uploaded_file = st.file_uploader(
                "Drag & drop atau klik untuk upload",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
                key="file_input",
            )
        else:
            uploaded_file = st.camera_input(
                "Ambil foto",
                label_visibility="collapsed",
                key="cam_input",
            )

        if uploaded_file is not None:
            img_bytes = uploaded_file.getvalue()
            img_sig = (len(img_bytes),)
            if st.session_state.uploaded_image_sig != img_sig:
                st.session_state.uploaded_image_bytes = img_bytes
                st.session_state.uploaded_image_sig = img_sig
                st.session_state.last_prediction_sig = None
                st.session_state.prediction_result = None

            image = Image.open(io.BytesIO(st.session_state.uploaded_image_bytes))
            st.markdown("<div class='section-header' style='margin-top:1rem;'>Preview</div>", unsafe_allow_html=True)
            st.image(image, use_container_width=True)

            if st.button("Hapus Gambar", use_container_width=True):
                st.session_state.uploaded_image_bytes = None
                st.session_state.uploaded_image_sig = None
                st.session_state.last_prediction_sig = None
                st.session_state.prediction_result = None
                st.rerun()
        else:
            st.markdown(
                "<div style='text-align:center; padding:3rem 1rem; color:#64748b;'>"
                "<p>Upload atau ambil foto buah untuk memulai analisis</p>"
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        has_image = st.session_state.get("uploaded_image_bytes") is not None
        if has_image:
            st.markdown(
                "<div class='glass-card animate-fade-in'>",
                unsafe_allow_html=True,
            )

            current_sig = st.session_state.uploaded_image_sig
            is_new_image = current_sig != st.session_state.last_prediction_sig

            if is_new_image or st.session_state.prediction_result is None:
                image = Image.open(io.BytesIO(st.session_state.uploaded_image_bytes))
                model = load_trained_model()
                if model is None:
                    st.error("Model tidak ditemukan atau gagal dimuat.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.stop()

                with st.spinner("Analisis AI..."):
                    processed_img = preprocess_image(image)
                    preds = model_predict(model, processed_img)
                    prob = tf.nn.softmax(preds[0]).numpy()
                    idx = int(np.argmax(prob))
                    confidence = float(prob[idx] * 100)
                    raw_name = CLASS_NAMES[idx]

                display_name = get_display_name(raw_name)
                clean_name = raw_name.split()[0]
                info = NUTRISI_DATA.get(clean_name)

                st.session_state.last_prediction_sig = current_sig
                st.session_state.prediction_result = {
                    "raw_name": raw_name,
                    "display_name": display_name,
                    "confidence": confidence,
                    "clean_name": clean_name,
                    "info": info,
                }
            else:
                result = st.session_state.prediction_result
                raw_name = result["raw_name"]
                display_name = result["display_name"]
                confidence = result["confidence"]
                clean_name = result["clean_name"]
                info = result["info"]

            color1, color2, shadow_color = get_fruit_color(display_name)
            st.markdown(
                f"""
                <div class='result-box animate-pulse' style='background: linear-gradient(135deg, {color1} 0%, {color2} 100%); box-shadow: 0 8px 25px {shadow_color};'>
                    <div class='result-label'>Buah Terdeteksi</div>
                    <div class='result-value'>{display_name}</div>
                    <div class='result-confidence'>Confidence: {confidence:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            bar_color = color1
            st.markdown(
                f"""
                <div style='margin: 1rem 0;'>
                    <div style='display:flex; justify-content:space-between; margin-bottom:0.5rem;'>
                        <span style='font-weight:600; color:#1e293b;'>Tingkat Keyakinan</span>
                        <span style='font-weight:700; color:{bar_color};'>{confidence:.1f}%</span>
                    </div>
                    <div class='confidence-bar-container'>
                        <div class='confidence-bar' style='width:{confidence}%; background:{bar_color};'></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<hr style='border:none; border-top:1px solid #e2e8f0; margin:1rem 0;'>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>Informasi Nutrisi (per 100g)</div>", unsafe_allow_html=True)

            if info:
                ncol1, ncol2 = st.columns(2)
                items = list(info.items())
                half = len(items) // 2 + len(items) % 2

                with ncol1:
                    for key, value in items[:half]:
                        icon = "" 
                        st.markdown(
                            f"""
                            <div class='nutrition-item'>
                                <div class='nutrition-label'>{key}</div>
                                <div class='nutrition-value'>{value}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                with ncol2:
                    for key, value in items[half:]:
                        icon = "" 
                        st.markdown(
                            f"""
                            <div class='nutrition-item'>
                                <div class='nutrition-label'>{key}</div>
                                <div class='nutrition-value'>{value}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
                fig_nutrition = create_nutrition_chart(info)
                st.plotly_chart(fig_nutrition, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Data nutrisi untuk buah ini belum tersedia dalam database.")

            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class='glass-card' style='text-align:center; padding:4rem 2rem;'>"
                "<h3 style='color:#1e293b; margin-bottom:0.5rem;'>Menunggu Input</h3>"
                "<p style='color:#64748b;'>Upload atau ambil foto buah di panel kiri untuk memulai analisis AI</p>"
                "</div>",
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS - Modern Glassmorphism Design
# ═══════════════════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-attachment: fixed;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Glass Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 1.5rem;
        box-shadow: 0 15px 35px -10px rgba(0, 0, 0, 0.2);
        margin-bottom: 1rem;
    }
    
    .glass-card-dark {
        background: rgba(30, 41, 59, 0.9);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        color: white;
    }
    
    /* Hero Title */
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.25rem;
    }
    
    .hero-subtitle {
        font-size: 0.95rem;
        color: #64748b;
        text-align: center;
        font-weight: 400;
        margin-bottom: 1rem;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    
    /* Result Display */
    .result-box {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border-radius: 16px;
        padding: 1rem 1.5rem;
        text-align: center;
        color: white;
        margin: 0.75rem 0;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.35);
    }
    
    .result-label {
        font-size: 0.75rem;
        font-weight: 500;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .result-value {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0.25rem 0;
    }
    
    .result-confidence {
        font-size: 0.95rem;
        font-weight: 600;
        opacity: 0.95;
    }
    
    /* Confidence Bar */
    .confidence-bar-container {
        background: #e2e8f0;
        border-radius: 10px;
        height: 12px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .confidence-bar {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease-out;
    }
    
    /* Nutrition Card */
    .nutrition-item {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 12px;
        padding: 0.6rem 0.9rem;
        margin: 0.35rem 0;
        border-left: 3px solid #667eea;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .nutrition-item:hover {
        transform: translateX(5px);
        box-shadow: 0 3px 10px rgba(102, 126, 234, 0.15);
    }
    
    .nutrition-label {
        font-size: 0.7rem;
        color: #64748b;
        font-weight: 500;
    }
    
    .nutrition-value {
        font-size: 0.95rem;
        color: #1e293b;
        font-weight: 700;
    }
    
    /* Top Predictions */
    .prediction-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.75rem 1rem;
        background: #f8fafc;
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    
    .prediction-rank {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
        color: white;
    }
    
    .rank-1 { background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%); }
    .rank-2 { background: linear-gradient(135deg, #94a3b8 0%, #64748b 100%); }
    .rank-3 { background: linear-gradient(135deg, #d97706 0%, #b45309 100%); }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: rgba(30, 41, 59, 0.95);
        backdrop-filter: blur(20px);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: white !important;
    }
    
    /* Upload Area */
    [data-testid="stFileUploader"] {
        background: rgba(102, 126, 234, 0.05);
        border: 2px dashed #667eea;
        border-radius: 16px;
        padding: 1rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }
    
    /* Radio Buttons - Force dark text */
    [data-testid="stRadio"] > div {
        background: #ffffff !important;
        border-radius: 12px;
        padding: 0.75rem 1.25rem;
        border: 1px solid #e2e8f0;
    }
    
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] label span,
    [data-testid="stRadio"] label p,
    [data-testid="stRadio"] div[data-testid="stMarkdownContainer"] p,
    [data-testid="stRadio"] * {
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    
    .stRadio > label {
        color: #1e293b !important;
    }
    
    div[role="radiogroup"] label {
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    
    div[role="radiogroup"] label div p {
        color: #1e293b !important;
    }
    
    /* Image Preview */
    [data-testid="stImage"] {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
    }
    
    /* Metric */
    [data-testid="stMetric"] {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 16px;
        padding: 1rem;
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .animate-fade-in {
        animation: fadeInUp 0.6s ease-out;
    }
    
    .animate-pulse {
        animation: pulse 2s infinite;
    }
    
    /* Divider */
    .fancy-divider {
        height: 3px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        border: none;
        margin: 1.5rem 0;
        border-radius: 2px;
    }
    
    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .stat-item {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 1.25rem;
        text-align: center;
        color: white;
    }
    
    .stat-value {
        font-size: 1.8rem;
        font-weight: 800;
    }
    
    .stat-label {
        font-size: 0.85rem;
        opacity: 0.9;
        margin-top: 0.25rem;
    }
</style>
"""

# Inject custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # SVG Icons (White Outline)
    icon_info = """<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"20\" height=\"20\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"white\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\" style=\"margin-right:8px; vertical-align:text-bottom;\"><circle cx=\"12\" cy=\"12\" r=\"10\"></circle><line x1=\"12\" y1=\"16\" x2=\"12\" y2=\"12\"></line><line x1=\"12\" y1=\"8\" x2=\"12.01\" y2=\"8\"></line></svg>"""
    icon_cpu = """<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"20\" height=\"20\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"white\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\" style=\"margin-right:8px; vertical-align:text-bottom;\"><rect x=\"4\" y=\"4\" width=\"16\" height=\"16\" rx=\"2\" ry=\"2\"></rect><rect x=\"9\" y=\"9\" width=\"6\" height=\"6\"></rect><line x1=\"9\" y1=\"1\" x2=\"9\" y2=\"4\"></line><line x1=\"15\" y1=\"1\" x2=\"15\" y2=\"4\"></line><line x1=\"9\" y1=\"20\" x2=\"9\" y2=\"23\"></line><line x1=\"15\" y1=\"20\" x2=\"15\" y2=\"23\"></line><line x1=\"20\" y1=\"9\" x2=\"23\" y2=\"9\"></line><line x1=\"20\" y1=\"14\" x2=\"23\" y2=\"14\"></line><line x1=\"1\" y1=\"9\" x2=\"4\" y2=\"9\"></line><line x1=\"1\" y1=\"14\" x2=\"4\" y2=\"14\"></line></svg>"""
    icon_book = """<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"20\" height=\"20\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"white\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\" style=\"margin-right:8px; vertical-align:text-bottom;\"><path d=\"M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z\"></path><path d=\"M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z\"></path></svg>"""

    st.markdown(f"<h3 style='color: white; margin-bottom:1.5rem; display:flex; align-items:center;'>{icon_info} Tentang Aplikasi</h3>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style='background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 12px; margin-bottom: 2rem;'>
            <p style='color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;'>
            Aplikasi ini membantu Anda mengenali jenis buah dan sayuran secara otomatis, serta memberikan informasi nutrisi yang bermanfaat untuk kesehatan Anda.
            </p>
        </div>
        """
        , unsafe_allow_html=True)

    st.markdown(f"<h4 style='color: white; font-size: 1rem; margin-bottom: 0.5rem; display:flex; align-items:center;'>{icon_cpu} Spesifikasi Model</h4>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style='color: #94a3b8; font-size: 0.85rem; margin-bottom: 2rem;'>
            <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 0.5rem;'>
                <span>Arsitektur</span>
                <span style='color: white;'>ResNet50</span>
            </div>
            <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 0.5rem;'>
                <span>Dataset</span>
                <span style='color: white;'>Fruits-360</span>
            </div>
            <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                <span>Kemampuan</span>
                <span style='color: white;'>50 Jenis Buah</span>
            </div>
        </div>
        """
        , unsafe_allow_html=True)

    st.markdown(f"<h4 style='color: white; font-size: 1rem; margin-bottom: 0.5rem; display:flex; align-items:center;'>{icon_book} Panduan Penggunaan</h4>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style='color: #cbd5e1; font-size: 0.85rem; line-height: 1.6;'>
            <p style='margin-bottom: 0.5rem;'>• Gunakan pencahayaan yang cukup</p>
            <p style='margin-bottom: 0.5rem;'>• Pastikan objek buah terlihat jelas</p>
            <p>• Background polos memberikan hasil terbaik</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; opacity:0.7; font-size:0.85rem;'>"
        "Made with ❤️ using Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
