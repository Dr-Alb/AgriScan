import os, io
from pathlib import Path
from flask import Flask, request, render_template_string, redirect, url_for, session
from openai import OpenAI
from twilio.rest import Client
import requests
from dotenv import load_dotenv
load_dotenv()
openai_client = OpenAI()          # reads key from env automatically

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


app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "defaultsecret")
DB_URL = "sqlite:///agriscan_users.db"
engine = create_engine(DB_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id       = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String)
    phone = Column(String)
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

# ─── API Keys ───
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_FROM")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

BASE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{{ title }}</title>
  <style>
    body {
      margin: 0;
      font-family: "Segoe UI", sans-serif;
      background: url('https://images.unsplash.com/photo-1581090700227-1e37b190418e?ixlib=rb-4.0.3&auto=format&fit=crop&w=1400&q=80') center/cover no-repeat fixed;
      color: #333;
    }
    .navbar {
      background-color: #28a745;
      color: white;
      padding: 10px 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .navbar a {
      color: white;
      margin-right: 15px;
      text-decoration: none;
    }
    .sidebar {
      height: 100vh;
      width: 200px;
      position: fixed;
      top: 0;
      left: 0;
      background-color: #222;
      padding-top: 60px;
    }
    .sidebar a {
      padding: 10px 15px;
      display: block;
      color: white;
      text-decoration: none;
    }
    .sidebar a:hover {
      background-color: #575757;
    }
    .main {
      margin-left: 220px;
      padding: 20px;
    }
    .footer {
      background-color: #222;
      color: white;
      text-align: center;
      padding: 15px;
      position: fixed;
      width: 100%;
      bottom: 0;
      left: 0;
    }
    .card {
      background: rgba(255, 255, 255, 0.85);
      padding: 20px;
      border-radius: 10px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.1);
      margin-bottom: 20px;
    }
    textarea, input, button {
      font-size: 1em;
      padding: 10px;
      margin-top: 10px;
    }
  </style>
</head>
<body>
  <div class="navbar">
    <div><strong>AgriScan AI</strong></div>
    <div>
      <a href="/">Home</a>
      <a href="/dashboard">Services</a>
      <a href="/chatbot">Chatbot</a>
      <a href="/login">Sign In</a>
      <a href="/signup">Sign Up</a>
    </div>
  </div>
  <div class="sidebar">
    <a href="/">Home</a>
    <a href="/dashboard">Services</a>
    <a href="/chatbot">Chatbot</a>
    <a href="/login">Sign In</a>
    <a href="/signup">Sign Up</a>
    <a href="/logout">Logout</a>
  </div>
  <div class="main">
    {{ body|safe }}
  </div>
  <div class="footer">
    © 2025 AgriScan AI. All rights reserved.
  </div>
</body>
</html>"""

# ─── Helper ───
def send_weather_sms(phone):
    try:
        weather = requests.get("https://wttr.in/Nairobi?format=3").text
        msg = f"🌤️ AgriScan Alert:\nToday's Weather: {weather}"
        twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)
        message = twilio_client.messages.create(body=msg, from_=TWILIO_FROM, to=phone)
        return True, f"Sent to {phone}"
    except Exception as e:
        return False, str(e)

def chat_with_gpt(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def page(title, body_html):
    return render_template_string(BASE_HTML, title=title, body=body_html)

# ─── Landing ──────────────────────────────────────────────────────────
LANDING_BODY = """
<header style="padding:80px 20px;border-radius:12px;color:#fff;background:url('https://images.unsplash.com/photo-1568605114967-8130f3a36994') center/cover;">
  <h1>Crop‑disease detection at your fingertips</h1>
  <p style="margin-top:14px;font-size:1.1rem;">Snap, upload &amp; save your harvest.</p>
  <a class="btn" href="{{ url_for('signup') }}">Get Started</a>
</header>
<section id="services" style="margin-top:60px;">
  <h2>Our Services</h2>
  <div class="card">
    <h3>Plant‑Disease Scan</h3>
    <p>Instant leaf‑disease diagnosis powered by AI.</p>
  </div>
</section>
"""
@app.route("/")
def landing():
    return page("AgriScan – Home", LANDING_BODY)

# ─── Sign‑Up ──────────────────────────────────────────────────────────
SIGNUP_BODY = """
   <div class='card'><h2>Create Account</h2>
    <form method="POST">
    <input name="username" placeholder="Username"><br>
    <input name="password" placeholder="Password" type="password"><br>
    <input name="phone" placeholder="Phone (+2547...)"><br>
    <button type="submit">Sign Up</button></form></div>
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
  <div class='card'><h2>Login</h2>
    <form method="POST">
    <input name="username" placeholder="Username"><br>
    <input name="password" placeholder="Password" type="password"><br>
    <button type="submit">Login</button></form></div>
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
<h2>Welcome, {{ user }} </h2>
<div class="card"><h3><a href="{{ url_for('scan') }}">Start Plant‑Disease Scan</a></h3></div>
<p style="margin-top:35px;"><a href="{{ url_for('logout') }}">Log out</a></p>
<p>Click below to send today's alert manually.</p>
    <a href='{{{{ url_for("send_alert") }}}}'><button>Send Weather Alert</button></a>
    <section id="services" class="mt-5">
  <h2>Our Services</h2>
  <ul>
    <li> Leaf Scanning - Detect plant health issues quickly</li>
    <li> Weather Alerts - Get daily weather forecasts via SMS</li>
    <li> AI Chatbot - Ask questions and get real-time advice</li>
  </ul>
</section>

    
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
<h2>Ask AgriScan AI 🤖</h2>
<div id="chatbox" style="max-width:600px;margin:40px auto;text-align:left;
     border:1px solid #ccc;padding:16px;border-radius:8px;min-height:280px;
     background:#fff;overflow-y:auto;"></div>

<form onsubmit="sendMsg();return false;" style="max-width:600px;margin:12px auto;">
  <input id="msg" style="width:80%;" placeholder="Type your question…"/>
  <button class="btn">Send</button>
</form>

<script>
async function sendMsg(){
  const box=document.getElementById('chatbox');
  const inp=document.getElementById('msg');
  const user=inp.value.trim(); if(!user) return;
  box.innerHTML+=`<p><b>You:</b> ${user}</p>`;
  inp.value=''; box.scrollTop=box.scrollHeight;
  const r = await fetch('{{ url_for("chat_api") }}',{
     method:'POST', headers:{'Content-Type':'application/json'},
     body: JSON.stringify({message:user})
  });
  const data = await r.json();
  box.innerHTML+=`<p style="color:var(--green);"><b>AgriScan AI:</b> ${data.reply}</p>`;
  box.scrollTop=box.scrollHeight;
}
</script>
"""

@app.route("/chat")
def chat():
    redir = _guard();     # reuse the login guard
    if redir: return redir
    return page("ChatBot", CHAT_BODY)
@app.route("/api/chat", methods=["POST"])
def chat_api():
    redir = _guard()
    if redir: return jsonify({"error":"login required"}), 401

    user_msg = request.json.get("message", "").strip()
    if not user_msg:
        return jsonify({"reply":"Ask me anything about crop health!"})

    completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system","content":
             "You are AgriScan AI, an agricultural assistant specialised in "
             "plant disease diagnosis and sustainable farming advice. "
             "Answer concisely (≤120 words) and, when relevant, suggest how to "
             "use the AgriScan leaf‑scanner."},
            {"role":"user","content":user_msg}
        ]
    )
    reply = completion.choices[0].message.content
    return jsonify({"reply": reply})


# ─── Health ping ──────────────────────────────────────────────────────
@app.route("/health")
def health(): return "OK", 200

@app.route("/send-alert", methods=["GET", "POST"])
def send_alert():
    db = Session()
    user = db.query(User).filter_by(username=session["user"]).first()
    db.close()
    if user and user.phone:
        success, msg = send_weather_sms(user.phone)
        return f"<h3>{msg}</h3><a href='{url_for('dashboard')}'>Back</a>"
    return "No phone found"

@app.route("/daily-job")
def daily_job():
    db = Session()
    users = db.query(User).filter(User.phone.isnot(None)).all()
    db.close()
    results = []
    for u in users:
        success, msg = send_weather_sms(u.phone)
        results.append((u.username, msg))
    log = "<br>".join([f"{u}: {msg}" for u, msg in results])
    return f"<h2>Daily SMS Alert Log</h2><p>{log}</p>", 200

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    response = ""
    if request.method == "POST":
        question = request.form["message"]
        response = chat_with_gpt(question)
    return render_template_string(BASE_HTML, title="ChatBot", body=f"""
    <div class='card'><h2>AgriScan AI ChatBot</h2>
    <form method="POST">
        <textarea name="message" placeholder="Ask me anything..." rows="4" cols="50"></textarea><br>
        <button type="submit">Ask</button>
    </form>
    <div style="margin-top: 20px;"><strong>Response:</strong><br>{response}</div></div>
    """)

# ─── Launch ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
