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

# ─── Flask App Initialization ───
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "defaultsecret")
DB_URL = "sqlite:///agriscan_users.db"
engine = create_engine(DB_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ─── OpenAI & Twilio Credentials ───
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


BASE_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{{ title }}</title>
<style>
/* Reset and base styles */
body {
    margin: 0;
    font-family: "Segoe UI", sans-serif;
    height: 100vh;
    overflow: hidden;
    position: relative;
    color: #fff;
}
/* Video background styling */
#bg-video {
    position: fixed;
    right: 0;
    bottom: 0;
    min-width: 100%;
    min-height: 100%;
    object-fit: cover;
    z-index: -1;
}

/* Overlay content styling */
.content {
    position: relative;
    z-index: 1;
    background: rgba(0,0,0,0.4);
    padding: 20px;
    height: 100%;
    overflow: auto;
}

/* Navbar styles */
.navbar {
    position: fixed;
    top: 0;
    width: 100%;
    background-color: #28a745;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 20px;
    z-index: 10;
}
.navbar a {
    color: #fff;
    margin-right: 15px;
    text-decoration: none;
    font-weight: bold;
}
.navbar .dropdown {
  position: relative; display: inline-block;
}
.dropdown-content {
  display: none; position: absolute; background-color: #28a745; min-width: 160px; box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2); z-index: 1;
}
.dropdown:hover .dropdown-content { display: block; }
.dropdown-content a {
  color: white;
}
"""
</body>
</html>

@app.route("/")
def landing():
    return render_template_string(BASE_HTML, title="Welcome", body="<h2>Welcome to AgriScan AI</h2><p>Smart farming starts here.</p>")

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
        prompt = request.form["prompt"]
        res = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        reply = res.choices[0].message.content

    return render_template_string(BASE_HTML, title="Chatbot", body=f"""
        <h2>Ask AgriScan AI</h2>
        <form method='POST'>
            <textarea name='prompt' rows='4' cols='50'></textarea><br>
            <button>Send</button>
        </form>
        <div class='card'><strong>Response:</strong><p>{reply}</p></div>
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
