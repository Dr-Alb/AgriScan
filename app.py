import os, io
from pathlib import Path
from flask import Flask, request, render_template_string, redirect, url_for, session, jsonify
from openai import OpenAI
from twilio.rest import Client
import requests
from dotenv import load_dotenv
import numpy as np
from PIL import Image
import tensorflow as tf
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker
from passlib.hash import bcrypt

load_dotenv()

# ─── Configuration ───
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "defaultsecret")
DB_URL = "sqlite:///agriscan_users.db"
engine = create_engine(DB_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ─── OpenAI & Twilio ───
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_FROM")

# ─── User Model ───
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String)
    phone = Column(String)
    hash = Column(String, nullable=False)

Base.metadata.create_all(bind=engine)

# ─── Load TFLite Model ───
MODEL_PATH = "plant_disease_model.tflite"
LABELS_PATH = "label_map.txt"

if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"{MODEL_PATH} not found")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
_in, _out = interpreter.get_input_details(), interpreter.get_output_details()
IMG_SZ = _in[0]["shape"][1]

with open(LABELS_PATH) as f:
    CLASS_NAMES = [l.strip() for l in f]

def predict_pil(img: Image.Image):
    arr = (np.array(img.convert("RGB").resize((IMG_SZ, IMG_SZ))) / 255.0).astype(np.float32)[None, ...]
    interpreter.set_tensor(_in[0]["index"], arr)
    interpreter.invoke()
    probs = interpreter.get_tensor(_out[0]["index"])[0]
    idx = int(np.argmax(probs))
    return {"class_": CLASS_NAMES[idx], "confidence": float(probs[idx])}

# ─── Base HTML Layout ───
BASE_HTML = """
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\"/>
  <title>{{ title }}</title>
  <style>
    body {
      margin: 0;
      font-family: \"Segoe UI\", sans-serif;
      background: url('https://images.unsplash.com/photo-1581090700227-1e37b190418e?ixlib=rb-4.0.3&auto=format&fit=crop&w=1400&q=80') center/cover no-repeat fixed;
      color: #333;
    }
    .navbar { background-color: #28a745; color: white; padding: 10px 20px; display: flex; justify-content: space-between; align-items: center; }
    .navbar a { color: white; margin-right: 15px; text-decoration: none; }
    .sidebar { height: 100vh; width: 200px; position: fixed; top: 0; left: 0; background-color: #222; padding-top: 60px; }
    .sidebar a { padding: 10px 15px; display: block; color: white; text-decoration: none; }
    .sidebar a:hover { background-color: #575757; }
    .main { margin-left: 220px; padding: 20px; }
    .footer { background-color: #222; color: white; text-align: center; padding: 15px; position: fixed; width: 100%; bottom: 0; left: 0; }
    .card { background: rgba(255, 255, 255, 0.85); padding: 20px; border-radius: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }
    textarea, input, button { font-size: 1em; padding: 10px; margin-top: 10px; }
  </style>
</head>
<body>
  <div class=\"navbar\">
    <div><strong>AgriScan AI</strong></div>
    <div>
      <a href=\"/\">Home</a>
      <a href=\"/dashboard\">Services</a>
      <a href=\"/chatbot\">Chatbot</a>
      <a href=\"/login\">Sign In</a>
      <a href=\"/signup\">Sign Up</a>
    </div>
  </div>
  <div class=\"sidebar\">
    <a href=\"/\">Home</a>
    <a href=\"/dashboard\">Services</a>
    <a href=\"/chatbot\">Chatbot</a>
    <a href=\"/login\">Sign In</a>
    <a href=\"/signup\">Sign Up</a>
    <a href=\"/logout\">Logout</a>
  </div>
  <div class=\"main\">
    {{ body|safe }}
  </div>
  <div class=\"footer\">
    © 2025 AgriScan AI. All rights reserved.
  </div>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(BASE_HTML, title="Welcome", body="<div class='card'><h2>Welcome to AgriScan AI</h2><p>Scan plant diseases, chat with our AI assistant, and receive smart agriculture alerts.</p></div>")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        phone = request.form.get("phone")
        session_db = SessionLocal()
        if session_db.query(User).filter_by(username=username).first():
            return "Username already exists."
        hashed = bcrypt.hash(password)
        new_user = User(username=username, password=password, phone=phone, hash=hashed)
        session_db.add(new_user)
        session_db.commit()
        session_db.close()
        return redirect(url_for("login"))
    form_html = """
    <div class='card'>
        <h2>Sign Up</h2>
        <form method='POST'>
            <input name='username' placeholder='Username' required><br>
            <input type='password' name='password' placeholder='Password' required><br>
            <input name='phone' placeholder='Phone Number'><br>
            <button type='submit'>Sign Up</button>
        </form>
    </div>"""
    return render_template_string(BASE_HTML, title="Sign Up", body=form_html)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        session_db = SessionLocal()
        user = session_db.query(User).filter_by(username=username).first()
        if user and bcrypt.verify(password, user.hash):
            session["user"] = username
            return redirect(url_for("dashboard"))
        return "Invalid credentials."
    form_html = """
    <div class='card'>
        <h2>Login</h2>
        <form method='POST'>
            <input name='username' placeholder='Username' required><br>
            <input type='password' name='password' placeholder='Password' required><br>
            <button type='submit'>Login</button>
        </form>
    </div>"""
    return render_template_string(BASE_HTML, title="Login", body=form_html)

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("home"))

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    username = session["user"]
    return render_template_string(BASE_HTML, title="Dashboard", body=f"<div class='card'><h2>Welcome, {username}!</h2><p>Choose a service from the sidebar.</p></div>")

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    response_text = ""
    if request.method == "POST":
        user_input = request.form.get("message")
        res = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": user_input}])
        response_text = res.choices[0].message.content
    form_html = f"""
    <div class='card'>
        <h2>Chat with Agri AI</h2>
        <form method='POST'>
            <textarea name='message' rows='4' placeholder='Ask me about plant diseases, weather, farming tips...'></textarea><br>
            <button type='submit'>Send</button>
        </form>
        <p><strong>Response:</strong><br>{response_text}</p>
    </div>"""
    return render_template_string(BASE_HTML, title="Chatbot", body=form_html)

@app.route("/scan", methods=["GET", "POST"])
def scan():
    result_html = ""
    if request.method == "POST":
        file = request.files.get("image")
        if file:
            img = Image.open(file.stream)
            result = predict_pil(img)
            result_html = f"""
            <div class='card'>
                <h3>Prediction Result</h3>
                <p><strong>Disease:</strong> {result['class_']}</p>
                <p><strong>Confidence:</strong> {result['confidence']:.2%}</p>
            </div>"""
    form_html = f"""
    <div class='card'>
        <h2>Scan Plant Disease</h2>
        <form method='POST' enctype='multipart/form-data'>
            <input type='file' name='image' accept='image/*' required><br>
            <button type='submit'>Scan</button>
        </form>
        {result_html}
    </div>"""
    return render_template_string(BASE_HTML, title="Scan", body=form_html)

@app.route("/health")
def health():
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
