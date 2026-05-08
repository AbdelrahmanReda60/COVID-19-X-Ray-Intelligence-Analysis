import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page Config (بقت أشيك)
# -----------------------------
st.set_page_config(
    page_title="AI X-Ray Scanner",
    page_icon="🩺",
    layout="wide"  # تحويل لـ Wide عشان نستغل المساحة
)

# إضافة CSS مخصص (اختياري بس بيفرق)
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    # تأكد من اسم الملف عندك
    model = tf.keras.models.load_model("final_covid_model.keras")
    return model

model = load_model()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2877/2877578.png", width=100) # أيقونة طبية
    st.title("About the App")
    st.info("""
    هذا النظام يستخدم الذكاء الاصطناعي لتحليل صور الأشعة السينية.
    
    **Instructions:**
    1. Upload a clear X-ray (JPG/PNG).
    2. Click 'Analyze'.
    3. Results appear instantly.
    """)
    st.warning("⚠️ Disclaimer: This is an AI tool, not a final medical diagnosis.")

# -----------------------------
# Main Title
# -----------------------------
st.title("🩺 COVID-19 X-Ray Intelligence Analysis")
st.write("---")

# -----------------------------
# Layout: Two Columns
# -----------------------------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📁 Upload Center")
    uploaded_file = st.file_uploader(
        "Choose an X-ray image...", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Current X-ray View", use_container_width=True)

with col2:
    st.subheader("🔍 Analysis Results")
    
    if uploaded_file is not None:
        if st.button("🚀 Analyze X-ray"):
            with st.spinner("Processing medical data..."):
                # Preprocessing
                img_array = image.resize((224, 224))
                img_array = np.array(img_array) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Prediction
                prediction = model.predict(img_array)[0][0]
                
                # Logic and Display
                st.write("---")
                if prediction > 0.5:
                    st.error("### Result: COVID Positive")
                    st.metric(label="Confidence Level", value=f"{prediction*100:.1f}%", delta="⚠️ Risk Found")
                else:
                    st.success("### Result: COVID Negative")
                    st.metric(label="Confidence Level", value=f"{(1-prediction)*100:.1f}%", delta="✅ Clear", delta_color="normal")
                
                # إضافة شريط تقدم (Progress Bar) للجمال
                st.progress(int(prediction * 100) if prediction > 0.5 else int((1-prediction)*100))
    else:
        st.info("Waiting for an image to be uploaded...")

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption("Developed with ❤️ using Streamlit & TensorFlow")