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
import logging
logging.basicConfig(level=logging.DEBUG)


load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "defaultsecret")
DB_URL = "sqlite:///agriscan_users.db"
engine = create_engine(DB_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_FROM")

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String)
    phone = Column(String)
    hash = Column(String, nullable=False)

Base.metadata.create_all(bind=engine)

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
      background: url('https://images.unsplash.com/photo-1692369584496-3216a88f94c1?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NzF8fGFncmljdWx0dXJlfGVufDB8fDB8fHww') center/cover no-repeat fixed;
      color: #333;
      position:relative;
      z-index: 0;
    }

.corner-video {
  position: fixed;
  bottom: 20px;
  left: 20px;
  width: 300px;
  height: 170px;
  object-fit: cover;
  border: 3px solid #fff;
  border-radius: 10px;
  box-shadow: 0 0 10px rgba(0,0,0,0.5);
  z-index: 5;
}
   
    .overlay {
      background: rgba(255, 191, 0, 0.5);
      padding: 30px;
      border-radius: 10px;
    }
    .main {
    margin-left: 220px;
    padding: 20px;
    position: relative;
    z-index: 1;
  }
    .navbar { background-color: #28a745; color: white; padding: 10px 20px; display: flex; justify-content: space-between; align-items: center; }
    .navbar a { color: white; margin-right: 15px; text-decoration: none; position: relative; }
    .navbar .dropdown:hover .dropdown-content { display: block; }
    .dropdown-content {
      display: none;
      position: absolute;
      background-color: #28a745;
      min-width: 160px;
      box-shadow: 0px 8px 16px rgba(0,0,0,0.2);
      z-index: 1;
    }

@media (max-width: 768px) {
  .sidebar {
    width: 100%;
    height: auto;
    position: relative;
  }

  .main {
    margin-left: 0;
    padding: 10px;
  }

  body {
    font-size: 16px;
  }

  .overlay {
    padding: 15px;
    background-color:#b57edc;
  }

  .dropdown-content {
    position: relative;
  }

  .video-bg {
    object-fit: contain;
  }
}
    
    .dropdown-content a {
      color: white;
      padding: 12px 16px;
      text-decoration: none;
      display: block;
    }
    .sidebar { height: 100vh; width: 200px; position: fixed; top: 0; left: 0; background-color: #222; padding-top: 60px; }
    .sidebar a { padding: 10px 15px; display: block; color: white; text-decoration: none; }
    .sidebar a:hover { background-color: #575757; }
    .main { margin-left: 220px; padding: 20px; }
    .footer {  background-color: #9b59b6; color: white; text-align: center; padding: 15px; position: fixed; width: 100%; bottom: 0; left: 0; }
    .card { background: rgba(255, 255, 255, 0.95); padding: 20px; border-radius: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.2); margin-bottom: 20px; }
    textarea, input, button { font-size: 1em; padding: 10px; margin-top: 10px; }
  </style>
</head>
<body>
  {% if title == "Welcome" %}
 <video class="corner-video" autoplay muted loop>
  <source src="https://www.pexels.com/video/workers-in-a-greenhouse-farming-checking-their-crops-3195396/">
  Your browser does not support the video tag.
</video>

  {% endif %}
  <div class="navbar">
    <div><strong>AgriScan AI</strong></div>
    <div>
    
  <div class="dropdown">
  <a href="/services">Services</a>
  <div class="dropdown-content">
    <a href="/dashboard">Leaf Scan</a>
    <a href="/send_alerts">Weather Alerts</a>
    <a href="/chatbot">Voice Chatbot</a>
  </div>

      </div>
      <a href="/login">Sign In</a>
      <a href="/signup">Sign Up</a>
    </div>
  </div>
  <div class="sidebar">
    <a href="/">Home</a>
    <a href="/dashboard">Leaf Scan</a>
    <a href="/send_alerts">Weather Alerts</a>
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
</html>
"""

@app.route("/services")
def services():
    content = """
    <h2>Our Services</h2>
    <div class='card'>
        <h3> Leaf Scan</h3>
        <p>Upload a photo of a plant leaf to detect possible diseases using AI. Receive instant diagnosis and suggestions.</p>
    </div>
    <div class='card'>
        <h3>🌦 Weather Alerts</h3>
        <p>We send you daily SMS weather updates using trusted weather sources. Stay prepared before planting or spraying.</p>
    </div>
    <div class='card'>
        <h3> Voice Chatbot</h3>
        <p>Talk to our AI assistant to get help with farming queries, crop suggestions, and more – using text or your voice.</p>
    </div>
    """
    return render_template_string(BASE_HTML, title="Services", body=content)


@app.route("/")
def landing():
    return render_template_string(BASE_HTML, title="Welcome", body="""
        <div class='overlay'>
            <h1>Welcome to AgriScan AI</h1>
            <p>Revolutionizing farming with artificial intelligence. Get instant leaf disease diagnosis, daily weather alerts, and voice-driven smart assistant for farmers!</p>
        </div>
    """)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        db = SessionLocal()
        username = request.form["username"]
        password = request.form["password"]
        phone = request.form.get("phone")

        if db.query(User).filter_by(username=username).first():
            return "Username already exists"

        hashed_pw = bcrypt.hash(password)
        user = User(username=username, password=password, phone=phone, hash=hashed_pw)
        db.add(user)
        db.commit()
        db.close()
        return redirect("/login")

    return render_template_string(BASE_HTML, title="Sign Up", body="""
        <h2>Create Account</h2>
        <form method='POST'>
            <input name='username' placeholder='Username' required><br>
            <input name='phone' placeholder='Phone Number' required><br>
            <input name='password' type='password' placeholder='Password' required><br>
            <button>Sign Up</button>
        </form>
    """)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        db = SessionLocal()
        username = request.form["username"]
        password = request.form["password"]
        user = db.query(User).filter_by(username=username).first()
        db.close()

        if user and bcrypt.verify(password, user.hash):
            session["user"] = username
            return redirect("/dashboard")
        else:
            return "Invalid credentials"

    return render_template_string(BASE_HTML, title="Sign In", body="""
        <h2>Sign In</h2>
        <form method='POST'>
            <input name='username' placeholder='Username' required><br>
            <input name='password' type='password' placeholder='Password' required><br>
            <button>Login</button>
        </form>
    """)

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user" not in session:
        return redirect("/login")

    body = """
        <h2>Leaf Scan</h2>
        <form method='POST' enctype='multipart/form-data'>
            <input type='file' name='image' accept='image/*' required><br>
            <button>Scan</button>
        </form>
    """

    if request.method == "POST":
        img = Image.open(request.files["image"])
        pred = predict_pil(img)
        result = f"Predicted: {pred['class_']} with {pred['confidence']*100:.2f}% confidence"
        body += f"<div class='card'><h3>Result</h3><p>{result}</p></div>"

    return render_template_string(BASE_HTML, title="Dashboard", body=body)

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    reply = ""
    if request.method == "POST":
        try:
            prompt = request.form["prompt"]
            res = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            reply = res.choices[0].message.content
        except Exception as e:
            reply = f"⚠️ Error: {str(e)}"

    return render_template_string(BASE_HTML, title="Chatbot", body=f"""
        <h2>Talk to AgriScan AI</h2>
        <form method='POST'>
            <textarea name='prompt' rows='3' cols='60' id='prompt'></textarea><br>
            <button>Send</button>
            <button type='button' onclick='startVoice()'>🎙 Speak</button>
        </form>
        <div class='card'><strong>Response:</strong><p id='reply'>{reply}</p></div>
        <script>
        function startVoice() {{
            const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
            recognition.lang = 'en-US';
            recognition.start();
            recognition.onresult = (event) => {{
                document.getElementById('prompt').value = event.results[0][0].transcript;
            }};
        }}
        if ('speechSynthesis' in window && "{reply}".length > 0) {{
            const utter = new SpeechSynthesisUtterance("{reply}");
            window.speechSynthesis.speak(utter);
        }}
        </script>
    """)


@app.route("/send_alerts")
def send_alerts():
    db = SessionLocal()
    users = db.query(User).all()
    for user in users:
        if user.phone:
            try:
                msg = requests.get("https://wttr.in/?format=3").text
                Client(TWILIO_SID, TWILIO_TOKEN).messages.create(
                    to=user.phone,
                    from_=TWILIO_FROM,
                    body=f"🌦 Weather Update: {msg}"
                )
            except Exception as e:
                print(f"Failed to send to {user.phone}: {e}")
    db.close()
    return "Weather alerts sent!"

@app.route("/health")
def health():
    return "OK", 200

from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime

def send_daily_weather_alerts():
    print(f"[{datetime.now()}] Sending weather alerts...")
    db = SessionLocal()
    users = db.query(User).all()
    for user in users:
        if user.phone:
            try:
                msg = requests.get("https://wttr.in/?format=3").text
                twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)
                twilio_client.messages.create(
                    to=user.phone,
                    from_=TWILIO_FROM,
                    body=f"🌦 Daily Weather Update: {msg}"
                )
                print(f"✔ Alert sent to {user.phone}")
            except Exception as e:
                print(f"❌ Failed for {user.phone}: {e}")
    db.close()

# Schedule it to run every day at 6:00 AM
scheduler = BackgroundScheduler()
scheduler.add_job(send_daily_weather_alerts, 'cron', hour=6, minute=0)
scheduler.start()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
