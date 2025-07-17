import os
from flask import Flask, request, render_template_string, redirect, url_for, session
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
from twilio.rest import Client
import requests

load_dotenv()

# ─── Config ───
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "defaultsecret")
Base = declarative_base()
engine = create_engine("sqlite:///agricscan.db")
Session = sessionmaker(bind=engine)

# ─── Twilio Config ───
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_TOKEN = os.getenv("TWILIO_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_FROM")

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
<html><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{{ title }}</title>
<style>
:root { --green: #0a8400; --dark: #044a00; --light: #e8ffe8; }
* { box-sizing: border-box; margin: 0; padding: 0; font-family: Arial, sans-serif; }
body { display: flex; flex-direction: column; min-height: 100vh; background: #f5fff5; color: #222; }
nav { display: flex; justify-content: space-between; align-items: center; padding: 14px 20px; background: var(--green); color: #fff; }
nav .brand { font-size: 1.4rem; font-weight: bold; }
nav a { color: #fff; text-decoration: none; margin-left: 20px; }
nav .hamburger { background: none; border: none; color: #fff; font-size: 1.4rem; cursor: pointer; }
#sidebar { position: fixed; top: 0; left: -260px; width: 260px; height: 100%; background: var(--dark); color: #fff; padding: 80px 20px 20px; transition: .3s; }
#sidebar a { display: block; color: #fff; text-decoration: none; margin: 12px 0; }
#sidebar.active { left: 0; }
main { flex: 1; padding: 60px 20px; text-align: center; }
footer { background: #ddd; padding: 12px; text-align: center; font-size: .9rem; color: #444; }
.btn, button { padding: 10px 22px; border: none; border-radius: 6px; background: var(--green); color: #fff; cursor: pointer; }
.card { display: inline-block; padding: 24px; margin: 14px; border-radius: 12px; min-width: 250px; background: #ffffffd0; }
input { padding: 10px; border: 1px solid #999; border-radius: 6px; width: 220px; }
header { padding: 300px 20px; border-radius: 12px; color: #fff;
  background: url('https://images.unsplash.com/photo-1692369584496-3216a88f94c1?q=80&w=1032&auto=format&fit=crop') center/cover; }
@media (max-width: 600px) {
  header { background: url('https://images.unsplash.com/photo-1615913783914-91fba0a7120f?crop=entropy&fit=crop&w=600&h=400') center/cover; padding: 160px 20px; }
}
</style>
<script>function toggleSidebar(){ document.getElementById('sidebar').classList.toggle('active'); }</script>
</head>
<body>
<nav>
  <button class="hamburger" onclick="toggleSidebar()">☰</button>
  <span class="brand">AgriScan</span>
  <div>
    <a href="#home">Home</a>
    <a href="#services">Services</a>
    <a href="#chatbot">ChatBot</a>
    {% if session.get('user') %}
      <a href="{{ url_for('dashboard') }}">Dashboard</a>
      <a href="{{ url_for('logout') }}">Logout</a>
    {% else %}
      <a href="{{ url_for('login') }}">Login</a>
      <a href="{{ url_for('signup') }}">Sign Up</a>
    {% endif %}
  </div>
</nav>
<div id="sidebar">
  <a href="{{ url_for('landing') }}">Home</a>
  <a href="#services">Services</a>
  <a href="#chatbot">ChatBot</a>
  {% if session.get('user') %}
    <a href="{{ url_for('dashboard') }}">Dashboard</a>
    <a href="{{ url_for('logout') }}">Logout</a>
  {% else %}
    <a href="{{ url_for('login') }}">Login</a>
    <a href="{{ url_for('signup') }}">Sign Up</a>
  {% endif %}
</div>
<main>{{ body|safe }}</main>
<footer>© 2025 AgriScan AI · Making farming smarter 🌱<br>Bringing solutions to you</footer>
</body></html>"""

# ─── Helper ───
def send_weather_sms(phone):
    try:
        weather = requests.get("https://wttr.in/Nairobi?format=3").text
        msg = f"🌤️ AgriScan Alert:\nToday's Weather: {weather}"
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        message = client.messages.create(body=msg, from_=TWILIO_FROM, to=phone)
        return True, f"Sent to {phone}"
    except Exception as e:
        return False, str(e)

# ─── Routes ─────
@app.route("/")
def landing():
    return render_template_string(BASE_HTML, title="Welcome", body="<header><h1>Welcome to AgriScan AI</h1></header>")

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
    <h2>Create Account</h2><form method="POST">
    <input name="username" placeholder="Username"><br><br>
    <input name="password" placeholder="Password" type="password"><br><br>
    <input name="phone" placeholder="Phone (+2547...)"><br><br>
    <button type="submit">Sign Up</button></form>
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
    <h2>Login</h2><form method="POST">
    <input name="username" placeholder="Username"><br><br>
    <input name="password" placeholder="Password" type="password"><br><br>
    <button type="submit">Login</button></form>
    """)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("landing"))

@app.route("/dashboard")
def dashboard():
    return render_template_string(BASE_HTML, title="Dashboard", body="""
    <h2>Welcome {{ session['user'] }}</h2>
    <p>Click below to send today's alert manually.</p>
    <a class="btn" href="{{ url_for('send_alert') }}">Send Weather Alert</a>
    """)

@app.route("/send-alert")
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

# ─── Run ───
if __name__ == "__main__":
    app.run(debug=True)
