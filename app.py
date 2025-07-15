import os
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf

# --- Flask app --------------------------------------------------------------
app = Flask(__name__)

# --- Load TFLite model & labels once ---------------------------------------
MODEL_PATH = "plant_disease_model.tflite"
LABEL_PATH = "label_map.txt"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(LABEL_PATH, "r") as f:
    CLASS_NAMES = [line.strip() for line in f]

IMG_SIZE = input_details[0]["shape"][1]  # assumes square input, e.g. 224

# --- Helper: run inference --------------------------------------------------
def predict_image(image: Image.Image):
    """Preprocess PIL image and run TFLite inference."""
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0           # normalize
    arr = arr[np.newaxis, ...]                              # add batch dim

    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]["index"])[0]

    top_idx = int(np.argmax(pred))
    return {
        "class": CLASS_NAMES[top_idx],
        "confidence": float(pred[top_idx])
    }

# --- Routes -----------------------------------------------------------------
@app.route("/health")
def health():
    return "OK", 200

# Simple HTML upload form (optional)
HTML_FORM = """
<!doctype html>
<title>AgriScan – Plant Disease Detector</title>
<h2>Upload a leaf image</h2>
<form method=post enctype=multipart/form-data action="/predict">
  <input type=file name=file accept="image/*">
  <input type=submit value="Analyze">
</form>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_FORM)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        result = predict_image(image)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Run locally ------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

