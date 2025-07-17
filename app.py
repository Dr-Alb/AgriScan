import os
from flask import Flask, request, render_template_string, redirect, url_for, session
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
from twilio.rest import Client
import requests
import openai

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
openai.api_key = os.getenv("OPENAI_API_KEY")

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
  <title>AgriScan AI Dashboard</title>
  <style>
    body {
      margin: 0;
      font-family: "Segoe UI", sans-serif;
    }

    header {
      background: url('https://images.unsplash.com/photo-1581090700227-1e37b190418e?ixlib=rb-4.0.3&auto=format&fit=crop&w=1400&q=80') center/cover;
      padding: 150px 20px;
      text-align: center;
      color: white;
    }

    h1 {
      font-size: 2.5em;
      margin-bottom: 10px;
    }

    .btn {
      background: #28a745;
      color: white;
      border: none;
      padding: 10px 20px;
      font-size: 1em;
      border-radius: 8px;
      cursor: pointer;
      transition: background 0.3s ease;
    }

    .btn:hover {
      background: #218838;
    }

    main {
      padding: 30px;
      max-width: 960px;
      margin: auto;
    }

    section {
      margin-bottom: 40px;
    }

    section h2 {
      font-size: 1.5em;
      border-bottom: 2px solid #28a745;
      padding-bottom: 5px;
      margin-bottom: 15px;
    }

    .card {
      background: #f4f4f4;
      padding: 20px;
      border-radius: 10px;
      margin-bottom: 20px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }

    @media (max-width: 600px) {
      header {
        background: url('https://images.unsplash.com/photo-1615913783914-91fba0a7120f?crop=entropy&fit=crop&w=600&h=400') center/cover;
        padding: 100px 20px;
      }

      h1 {
        font-size: 1.8em;
      }

      .btn {
        width: 100%;
        font-size: 1em;
      }
    }
  </style>
</head>
<body>

  <header>
    <h1>Welcome</h1>
    <p>Your AgriScan AI Dashboard</p>
    <form action="{{ url_for('send_alert') }}" method="POST">
      <button class="btn" type="submit">Send Today's Weather Alert</button>
    </form>
  </header>

  <main>
    <!-- Services Section -->
    <section>
      <h2>Our Services</h2>
      <div class="card">
        <p><strong> Weather Alerts:</strong> Receive timely weather notifications to help you plan your planting and spraying effectively.</p>
      </div>
      <div class="card">
        <p><strong> AI Crop Advisor:</strong> Use our chatbot to ask farming questions and get instant help powered by AI.</p>
      </div>
      <div class="card">
        <p><strong> Smart Reports:</strong> Analyze past alerts and forecast trends to optimize your yield season after season.</p>
      </div>
    </section>

    <!-- Optional Future Section -->
    <section>
      <h2>Quick Access</h2>
      <ul>
        <li><a href="/chatbot">Talk to AgriChat Assistant</a></li>
        <li><a href="/alerts">View Alert History</a></li>
        <li><a href="/logout">Logout</a></li>
      </ul>
    </section>
  </main>

</body>
</html>


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

def chat_with_gpt(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{ "role": "user", "content": prompt }]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# ─── Routes ───
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
    return render_template_string(BASE_HTML, title="Dashboard", body=f"""
    <h2>Welcome {session['user']}</h2>
    <p>Click below to send today's alert manually.</p>
    <a class="btn" href="{{{{ url_for('send_alert') }}}}">Send Weather Alert</a>
    <hr><h3 id="services"> Services</h3>
    <div class="card"><h4> SMS Alerts</h4><p>Daily weather notifications via SMS</p></div>
    <div class="card"><h4> ChatBot</h4><p>Ask the AI about crops, pests, climate, etc.</p></div>
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

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    response = ""
    if request.method == "POST":
        question = request.form["message"]
        response = chat_with_gpt(question)
    return render_template_string(BASE_HTML, title="ChatBot", body=f"""
    <h2>AgriScan AI ChatBot</h2>
    <form method="POST">
        <textarea name="message" placeholder="Ask me anything..." rows="4" cols="50"></textarea><br><br>
        <button type="submit">Ask</button>
    </form>
    <div style="margin-top: 20px;"><strong>Response:</strong><br>{response}</div>
    """)

# ─── Run ───
if __name__ == "__main__":
    app.run(debug=True)
