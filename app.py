import os
import streamlit as st
import numpy as np
from flask import Flask
from PIL import Image
import tensorflow as tf
from supabase import create_client, Client

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello from Render!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # default to 5000 for local dev
    app.run(host="0.0.0.0", port=port)

# ---- Config ----
st.set_page_config(page_title="AgriScanAI 🌿", layout="centered")

# ---- Supabase connection (use Streamlit secrets) ----
SUPABASE_URL: str = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY: str = st.secrets.get("SUPABASE_KEY", "")

sb: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---- Load labels & model ----
with open("label_map.txt") as f:
    LABELS = f.read().splitlines()

@st.cache_resource
def load_tflite():
    interpreter = tf.lite.Interpreter(model_path="plant_disease_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---- Helper functions ----
def preprocess(img: Image.Image):
    img = img.resize((224, 224))
    arr = np.array(img).astype(np.float32)
    return np.expand_dims(arr, 0)

def predict(img):
    inp = preprocess(img)
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    idx = int(np.argmax(output))
    conf = float(output[idx])
    return LABELS[idx], conf

def log_to_supabase(label, conf):
    if sb:
        try:
            sb.table("predictions").insert({"label": label, "confidence": conf}).execute()
        except Exception as e:
            st.warning(f"Could not log to Supabase: {e}")

# ---- UI ----
st.markdown("""<h1 style='text-align:center;color:#2E7D32;'>🌿 AgriScanAI</h1>""", unsafe_allow_html=True)
st.write("Upload a leaf image or use your camera to detect plant diseases. Results can be logged to Supabase if configured.")

file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
if not file:
    file = st.camera_input("Or take a photo")

if file:
    img = Image.open(file)
    st.image(img, caption="Input Image", use_column_width=True)
    with st.spinner("Analyzing..."):
        label, conf = predict(img)
    st.success(f"**Prediction:** {label}")
    st.info(f"Confidence: {conf*100:.2f}%")
    if sb:
        log_to_supabase(label, conf)
        st.success("Logged to Supabase ✅")
    else:
        st.info("Supabase not configured (add SUPABASE_URL & SUPABASE_KEY in Streamlit secrets)")

st.markdown("""<p style='text-align:center;color:grey;'>Powered by TensorFlow Lite, Streamlit & Supabase</p>""", unsafe_allow_html=True)
