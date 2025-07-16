import os, io
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)

import numpy as np
from PIL import Image
from flask import (
    Flask, render_template_string, request, redirect,
    url_for, session, jsonify
)

# ─── Tiny ORM layer (SQLAlchemy + SQLite) ─────────────────────────────
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker
from passlib.hash import bcrypt

DB_URL = "sqlite:///agriscan_users.db"
engine = create_engine(DB_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id       = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    hash     = Column(String,  nullable=False)

Base.metadata.create_all(bind=engine)

# ─── TFLite model load ────────────────────────────────────────────────
MODEL_PATH  = "plant_disease_model.tflite"
LABELS_PATH = "label_map.txt"

if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"  {MODEL_PATH} not found")

import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
_in, _out = interpreter.get_input_details(), interpreter.get_output_details()
IMG_SZ = _in[0]["shape"][1]

with open(LABELS_PATH) as f:
    CLASS_NAMES = [l.strip() for l in f]

def predict_pil(img: Image.Image):
    arr = (np.array(img.convert("RGB").resize((IMG_SZ, IMG_SZ))) / 255.0
           ).astype(np.float32)[None, ...]
    interpreter.set_tensor(_in[0]["index"], arr)
    interpreter.invoke()
    probs = interpreter.get_tensor(_out[0]["index"])[0]
    idx   = int(np.argmax(probs))
    return {"class_": CLASS_NAMES[idx], "confidence": float(probs[idx])}

# ─── Flask app & layout template ──────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "CHANGE_ME_IN_PROD")

BASE_HTML = """
<!doctype html><html><head>
  <title>{{ title or 'AgriScan' }}</title>
  <meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
  <style>
    :root{--green:#0a8400;--dark:#044a00;--light:#e8ffe8}
    *{box-sizing:border-box;margin:0;padding:0;font-family:Arial}
    body{min-height:100vh;display:flex;flex-direction:column;background:#f5fff5;color:#222}
    /* NAVBAR */
    nav{display:flex;justify-content:space-between;align-items:center;
        background:var(--green);color:#fff;padding:14px 20px}
    nav .brand{font-size:1.4rem;font-weight:bold}
    nav a{color:#fff;text-decoration:none;margin-left:20px}
    nav .hamburger{font-size:1.4rem;background:none;border:none;color:#fff;cursor:pointer}
    /* SIDEBAR */
    #sidebar{position:fixed;top:0;left:-260px;width:260px;height:100%;
             background:var(--dark);padding:80px 20px 20px;color:#fff;transition:.3s}
    #sidebar a{display:block;color:#fff;text-decoration:none;margin:12px 0}
    #sidebar.active{left:0}
    /* MAIN + FOOTER */
    main{flex:1;padding:60px 20px;text-align:center}
    footer{background:#ddd;text-align:center;padding:12px;font-size:.9rem;color:#444}
    /* UI helpers */
    .btn,button{padding:10px 22px;border:none;border-radius:6px;background:var(--green);color:#fff;cursor:pointer}
    .card{background:#ffffffd0;border-radius:12px;display:inline-block;padding:24px;margin:14px;min-width:250px}
    input{padding:10px;border-radius:6px;border:1px solid #999;width:220px}
  </style>
  <script>function toggleSidebar(){document.getElementById('sidebar').classList.toggle('active');}</script>
</head><body>
<nav>
  <button class="hamburger" onclick="toggleSidebar()">☰</button>
  <span class="brand">AgriScan</span>
  <div>
   <a href="{{ url_for('chat') }}">ChatBot</a>
    <a href="{{ url_for('landing') }}">Home</a>
    <a href="{{ url_for('landing') }}#services">Services</a>
    {% if not session.get('user') %}
      <a href="{{ url_for('login') }}">Login</a>
      <a href="{{ url_for('signup') }}">Sign Up</a>
    {% else %}
      <a href="{{ url_for('dashboard') }}">Dashboard</a>
      <a href="{{ url_for('logout') }}">Logout</a>
    {% endif %}
  </div>
</nav>
<div id="sidebar">
 <a href="{{ url_for('chat') }}">ChatBot</a>
  <a href="{{ url_for('landing') }}"> Home</a>
  <a href="{{ url_for('landing') }}#services">🛠 Services</a>
  {% if not session.get('user') %}
    <a href="{{ url_for('login') }}"> Login</a>
    <a href="{{ url_for('signup') }}"> Sign Up</a>
  {% else %}
    <a href="{{ url_for('dashboard') }}"> Dashboard</a>
    <a href="{{ url_for('logout') }}">Logout</a>
  {% endif %}
</div>
<main>{{ body | safe }}</main>
<footer>© 2025 AgriScan AI · Making farming smarter 🌱
No specialist required zero cost 100% harvest</footer>
</body></html>
"""

def page(title, body_html):
    return render_template_string(BASE_HTML, title=title, body=body_html)

# ─── Landing ──────────────────────────────────────────────────────────
LANDING_BODY = """
<header style="padding:80px 20px;border-radius:12px;color:#fff;background:url('https://images.unsplash.com/photo-1568605114967-8130f3a36994') center/cover;">
  <h1>Crop‑disease detection at your fingertips</h1>
  <p style="margin-top:18px; margin-bottom:18px;font-size:1.1rem;">Snap, upload &amp; save your harvest.</p><p>No need to pay a specialist</p>
  <a class="btn" href="{{ url_for('signup') }}">Get Started</a>
</header>
<section id="services" style="margin-top:60px;">
  <h2>Our Services</h2>
  <div class="card">
    <h3>Plant‑Disease Scan</h3>
    <p>Instant leaf‑disease diagnosis powered by AI.</p>
    <p>Just a snap of your crop leafe and you save your harvest</p>
  </div>
</section>
"""
@app.route("/")
def landing():
    return page("AgriScan – Home", LANDING_BODY)

# ─── Sign‑Up ──────────────────────────────────────────────────────────
SIGNUP_BODY = """
<h2>Create an account</h2>
<form method="post" style="margin-top:40px;">
  <input type="text" name="u" placeholder="Choose username"><br><br>
  <input type="password" name="p" placeholder="Choose password"><br><br>
  <button>Sign Up</button>
</form>
{% if error %}<p style="color:red;margin-top:14px;">{{ error }}</p>{% endif %}
<p style="margin-top:22px;">Already have an account? <a href="{{ url_for('login') }}">Log in</a></p>
"""
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        u = request.form["u"].strip().lower()
        p = request.form["p"]
        if not u or not p:
            return page("Sign Up", render_template_string(SIGNUP_BODY, error="All fields required"))
        db = SessionLocal()
        if db.query(User).filter_by(username=u).first():
            db.close()
            return page("Sign Up", render_template_string(SIGNUP_BODY, error="Username already taken"))
        db.add(User(username=u, hash=bcrypt.hash(p)))
        db.commit(); db.close()
        session["user"] = u
        return redirect(url_for("dashboard"))
    return page("Sign Up", render_template_string(SIGNUP_BODY, error=None))

# ─── Login ────────────────────────────────────────────────────────────
LOGIN_BODY = """
<h2>Login</h2>
<form method="post" style="margin-top:40px;">
  <input type="text" name="u" placeholder="Username"><br><br>
  <input type="password" name="p" placeholder="Password"><br><br>
  <button>Login</button>
</form>
{% if error %}<p style="color:red;margin-top:14px;">{{ error }}</p>{% endif %}
<p style="margin-top:22px;">New here? <a href="{{ url_for('signup') }}">Create an account</a></p>
"""
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form["u"].strip().lower()
        p = request.form["p"]
        db = SessionLocal()
        user = db.query(User).filter_by(username=u).first(); db.close()
        if user and bcrypt.verify(p, user.hash):
            session["user"] = u
            return redirect(url_for("dashboard"))
        return page("Login", render_template_string(LOGIN_BODY, error="Invalid credentials"))
    return page("Login", render_template_string(LOGIN_BODY, error=None))

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("landing"))

def _guard():
    if "user" not in session:
        return redirect(url_for("login"))

# ─── Dashboard ────────────────────────────────────────────────────────
DASHBOARD_BODY = """
<h2>Welcome, Bringing solutions to you {{ user }} </h2>
<div class="card"><h3><a href="{{ url_for('scan') }}">Start Plant‑Disease Scan</a></h3></div>
<p style="margin-top:35px;"><a href="{{ url_for('logout') }}">Log out</a></p>
"""
@app.route("/dashboard")
def dashboard():
    redir = _guard();   # returns redirect if not logged in
    if redir: return redir
    return page("Dashboard", render_template_string(DASHBOARD_BODY, user=session["user"]))

# ─── Scan + Predict ───────────────────────────────────────────────────
SCAN_BODY = """
<h2>Upload Leaf Image</h2>
<form action="{{ url_for('predict') }}" method="post" enctype="multipart/form-data" style="margin-top:40px;">
  <input type="file" name="file" accept="image/*"><br><br>
  <button>Scan</button>
</form>
{% if result %}
  <h3 style="margin-top:40px;">Result</h3>
  <p>Disease/Status: <b>{{ result.class_ }}</b></p>
  <p>Confidence score: {{ '{:.1%}'.format(result.confidence) }}</p>
{% endif %}
<p style="margin-top:30px;"><a href="{{ url_for('dashboard') }}">⬅ Back to dashboard</a></p>
"""
@app.route("/scan")
def scan():
    redir = _guard();  # must be logged in
    if redir: return redir
    return page("Leaf Scan", render_template_string(SCAN_BODY, result=None))

@app.route("/predict", methods=["POST"])
def predict():
    redir = _guard()
    if redir: return redir
    if "file" not in request.files or request.files["file"].filename == "":
        return redirect(url_for("scan"))
    try:
        img = Image.open(io.BytesIO(request.files["file"].read()))
        result = predict_pil(img)
        return page("Leaf Scan", render_template_string(SCAN_BODY, result=result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

CHAT_BODY = """
<h2>Ask AgriScan AI </h2>
<div id="chatbox" style="max-width:600px;margin:40px auto;text-align:left;
     border:1px solid #ccc;padding:16px;border-radius:8px;min-height:280px;
     background:#fff;overflow-y:auto;"></div>

<form onsubmit="sendMsg();return false;" style="max-width:600px;margin:12px auto;">
  <input id="msg" style="width:80%;" placeholder="Type your question…"/>
  <button class="btn">Send</button>
</form>

<script>
async function sendMessage() {
    const userInput = document.getElementById("userMessage").value;
    const chatBox = document.getElementById("chatBox");

    chatBox.innerHTML += `<div><strong>You:</strong> ${userInput}</div>`;

    const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userInput })
    });

    const data = await response.json();
    if (data.response) {
        chatBox.innerHTML += `<div><strong>Bot:</strong> ${data.response}</div>`;
    } else {
        chatBox.innerHTML += `<div><strong>Bot:</strong> Error: ${data.error}</div>`;
    }

    document.getElementById("userMessage").value = "";
}
</script>

"""

@app.route("/chat")
def chat():
    redir = _guard();     # reuse the login guard
    if redir: return redir
    return page("ChatBot", CHAT_BODY)
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    prompt = data.get("message")
    if not prompt:
        return jsonify({"error": "No message provided"}), 400

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # or gpt-4 if you have access
            messages=[{"role": "user", "content": prompt}]
        )
        reply = response.choices[0].message.content
        return jsonify({"response": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── Health ping ──────────────────────────────────────────────────────
@app.route("/health")
def health(): return "OK", 200

# ─── Launch ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
