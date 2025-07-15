import os, io
import numpy as np
from db import SessionLocal, User          # tiny ORM helpers
from passlib.hash import bcrypt

# --- replace SECRET_KEY with env default if not done ---
app.secret_key = os.getenv("SECRET_KEY", "change_me")

from PIL import Image
from flask import (
    Flask, request, jsonify,
    render_template_string, redirect, url_for, session
)
import tensorflow as tf

# ── Flask & session ────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change_me")  # replace in prod

# ── Load TFLite model once (lightweight) ───────────────────────────────
MODEL_PATH  = "plant_disease_model.tflite"
LABELS_PATH = "label_map.txt"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
in_det  = interpreter.get_input_details()
out_det = interpreter.get_output_details()
IMG_SZ  = in_det[0]["shape"][1]

with open(LABELS_PATH) as f:
    CLASS_NAMES = [l.strip() for l in f]

def predict_pil(img: Image.Image):
    arr = (np.array(img.convert("RGB").resize((IMG_SZ, IMG_SZ))) / 255.0
           ).astype(np.float32)[np.newaxis, ...]
    interpreter.set_tensor(in_det[0]["index"], arr)
    interpreter.invoke()
    pred = interpreter.get_tensor(out_det[0]["index"])[0]
    idx  = int(np.argmax(pred))
    return dict(class_=CLASS_NAMES[idx], confidence=float(pred[idx]))

# ── HTML snippets (inline for brevity) ─────────────────────────────────
BASE_CSS = """
<style>
 body {margin:0;font-family:Arial;color:#fff;text-align:center;}
 header {background:#004d00aa;padding:80px 20px;}
 section {padding:60px 20px;}
 .card {background:#ffffff22;border-radius:12px;padding:25px;display:inline-block;}
 input,button{padding:10px;border-radius:8px;border:none}
 a.btn{background:#0a0;color:#fff;text-decoration:none;padding:10px 25px;border-radius:8px}
</style>
"""

LANDING = BASE_CSS + """
<body style="background:url('https://images.unsplash.com/photo-1568605114967-8130f3a36994') center/cover;">
<header>
  <h1>🌿 AgriScan AI</h1>
  <p>Your pocket‑assistant for early crop‑disease detection.</p>
  <a class="btn" href="{{ url_for('login') }}">Login to continue</a>
</header>

<section>
  <h2>Our Services</h2>
  <div class="card">
    <h3>Plant‑Disease Scan</h3>
    <p>Upload a photo of a leaf and get an instant diagnosis.</p>
  </div>
  <!-- Add more service cards later -->
</section>
</body>
"""

LOGIN_PAGE = BASE_CSS + """
<body style="background:#e8ffe8;">
<header><h2>Log in to AgriScan</h2></header>
<section>
  <form method="post">
    <input type="text"     name="u" placeholder="Username"><br><br>
    <input type="password" name="p" placeholder="Password"><br><br>
    <button>Login</button>
  </form>
  {% if error %}<p style="color:#ff0">{{ error }}</p>{% endif %}
  <p>No account? <a href="{{ url_for('signup') }}">Sign up</a></p>
</section>
</body>
"""
DASHBOARD = BASE_CSS + """
<body style="background:#f0fff0;">
<header><h2>Welcome, {{ user }} 👋</h2></header>
<section>
  <div class="card">
    <h3><a href="{{ url_for('scan') }}">👉 Start Plant‑Disease Scan</a></h3>
  </div>
  <p><a href="{{ url_for('logout') }}">Log out</a></p>
</section>
</body>
"""

SCAN_FORM = BASE_CSS + """
<body style="background:#f5fff5;">
<header><h2>Upload Leaf Image</h2></header>
<section>
  <form action="{{ url_for('predict') }}" method="post" enctype="multipart/form-data">
    <input type="file" name="file" accept="image/*"><br><br>
    <button>Scan</button>
  </form>
  {% if result %}
     <h3>Result</h3>
     <p>Disease/Status: <b>{{ result.class_ }}</b></p>
     <p>Confidence: {{ '{:.1%}'.format(result.confidence) }}</p>
  {% endif %}
  <p><a href="{{ url_for('dashboard') }}">⬅ Back</a></p>
</section>
</body>
"""

SIGNUP_PAGE = BASE_CSS + """
<body style="background:#e8ffe8;">
<header><h2>Create a new AgriScan account</h2></header>
<section>
  <form method="post">
    <input type="text"     name="u" placeholder="Choose username"><br><br>
    <input type="password" name="p" placeholder="Choose password"><br><br>
    <button>Sign Up</button>
  </form>
  {% if error %}<p style="color:#ff0">{{ error }}</p>{% endif %}
  <p>Already have an account? <a href="{{ url_for('login') }}">Log in</a></p>
</section>
</body>
"""

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["u"].strip().lower()
        password = request.form["p"]

        if not username or not password:
            return render_template_string(SIGNUP_PAGE, error="All fields required")

        db = SessionLocal()
        if db.query(User).filter_by(username=username).first():
            db.close()
            return render_template_string(SIGNUP_PAGE, error="Username already taken")

        db.add(User(username=username, hash=bcrypt.hash(password)))
        db.commit()
        db.close()
        # auto‑login after sign‑up
        session["user"] = username
        return redirect(url_for("dashboard"))

    return render_template_string(SIGNUP_PAGE, error=None)

# ── Routes ─────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return "OK", 200

@app.route("/")
def landing():
    return render_template_string(LANDING)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["u"].strip().lower()
        password = request.form["p"]

        db = SessionLocal()
        user = db.query(User).filter_by(username=username).first()
        db.close()

        if user and bcrypt.verify(password, user.hash):
            session["user"] = username
            return redirect(url_for("dashboard"))

        return render_template_string(LOGIN_PAGE, error="Invalid credentials")

    return render_template_string(LOGIN_PAGE, error=None)


        db.add(User(username=username, hash=bcrypt.hash(password)))
        db.commit()
        db.close()
        # auto‑login after sign‑up
        session["user"] = username
        return redirect(url_for("dashboard"))

    return render_template_string(SIGNUP_PAGE, error=None)
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("landing"))

def _need_login():
    if "user" not in session:
        return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
    chk = _need_login()
    if chk: return chk
    return render_template_string(DASHBOARD, user=session["user"])

@app.route("/scan", methods=["GET"])
def scan():
    chk = _need_login()
    if chk: return chk
    return render_template_string(SCAN_FORM, result=None)

@app.route("/predict", methods=["POST"])
def predict():
    chk = _need_login()
    if chk: return chk
    if "file" not in request.files or request.files["file"].filename == "":
        return redirect(url_for("scan"))

    try:
        img = Image.open(io.BytesIO(request.files["file"].read()))
        result = predict_pil(img)
        return render_template_string(SCAN_FORM, result=result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Run locally ────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
