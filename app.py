import os
from flask import Flask, request, render_template_string, redirect, url_for, session
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
from twilio.rest import Client
import requests
from openai import OpenAI

load_dotenv()

# ─── Config ───
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "defaultsecret")
Base = declarative_base()
engine = create_engine("sqlite:///agricscan.db")
Session = sessionmaker(bind=engine)

# ─── API Keys ───
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_FROM")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# ─── Models ──
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    password = Column(String)
    phone = Column(String)

Base.metadata.create_all(engine)

# ─── Templates ──
BASE_HTML = """<!doctype html>
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

# ─── Routes ───
@app.route("/send-alert", methods=["GET", "POST"])
def landing():
    return render_template_string(BASE_HTML, title="Home", body="""
    <div class='card'><h2>Welcome to AgriScan AI</h2>
    <p>Your AI-powered agricultural assistant.</p>
    <form action='{{ url_for("send_alert") }}' method=(['POST'"POST"])>
      <button type='submit'>Send Today's Weather Alert</button>
    </form></div>
    """)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        db = Session()
        user = User(username=request.form["username"], password=request.form["password"], phone=request.form["phone"])
        db.add(user)
        db.commit()
        db.close()
        return redirect(url_for("login"))
    return render_template_string(BASE_HTML, title="Sign Up", body="""
    <div class='card'><h2>Create Account</h2>
    <form method="POST">
    <input name="username" placeholder="Username"><br>
    <input name="password" placeholder="Password" type="password"><br>
    <input name="phone" placeholder="Phone (+2547...)"><br>
    <button type="submit">Sign Up</button></form></div>
    """)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        db = Session()
        user = db.query(User).filter_by(username=request.form["username"], password=request.form["password"]).first()
        db.close()
        if user:
            session["user"] = user.username
            return redirect(url_for("dashboard"))
        return "Invalid login"
    return render_template_string(BASE_HTML, title="Login", body="""
    <div class='card'><h2>Login</h2>
    <form method="POST">
    <input name="username" placeholder="Username"><br>
    <input name="password" placeholder="Password" type="password"><br>
    <button type="submit">Login</button></form></div>
    """)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("landing"))

@app.route("/dashboard")
def dashboard():
    return render_template_string(BASE_HTML, title="Dashboard", body=f"""
    <div class='card'><h2>Welcome {session.get('user', '')}</h2>
    <p>Click below to send today's alert manually.</p>
    <a href='{{{{ url_for("send_alert") }}}}'><button>Send Weather Alert</button></a>
    <hr>
    <h3>Our Services</h3>
    <ul>
      <li>leafe scanning for diseases</li>
      <li>SMS Alerts for daillly weather updates</li>
      <li>ChatBot Support to give you solutiond on your crops</li>
      <li>Smart Forecasting of the best weather patterns</li>
    </ul></div>
    """)

@app.route("/send-alert", methods=["POST"])
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

# ─── Run ───
if __name__ == "__main__":
    app.run(debug=True)
