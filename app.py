# ──────────────────────────  AgriScan AI  ────────────────────────────
import os, io
from pathlib import Path

from dotenv            import load_dotenv
from openai            import OpenAI
from PIL               import Image
import numpy as np
import tensorflow as tf

from flask import (
    Flask, request, session, redirect, url_for,
    jsonify, render_template_string
)

# ─── Load secrets ────────────────────────────────────────────────────
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─── DB (SQLAlchemy + SQLite) ─────────────────────────────────────────
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker
from passlib.hash import bcrypt

DB_URL = "sqlite:///agriscan_users.db"
engine  = create_engine(DB_URL, echo=False, connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)
Base    = declarative_base()

class User(Base):
    __tablename__ = "users"
    id       = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    hash     = Column(String,  nullable=False)

Base.metadata.create_all(bind=engine)

# ─── TFLite model ────────────────────────────────────────────────────
MODEL_PATH  = "plant_disease_model.tflite"
LABELS_PATH = "label_map.txt"

if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"{MODEL_PATH} missing")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
_in, _out = interpreter.get_input_details(), interpreter.get_output_details()
IMG_SZ = _in[0]["shape"][1]

CLASS_NAMES = [l.strip() for l in open(LABELS_PATH)]

def predict_pil(img: Image.Image):
    arr = (np.array(img.convert("RGB").resize((IMG_SZ, IMG_SZ))) / 255.0
           ).astype(np.float32)[None, ...]
    interpreter.set_tensor(_in[0]["index"], arr)
    interpreter.invoke()
    probs = interpreter.get_tensor(_out[0]["index"])[0]
    idx   = int(np.argmax(probs))
    return {"class_": CLASS_NAMES[idx], "confidence": float(probs[idx])}

# ─── Flask app & helpers ─────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "CHANGE_ME_IN_PROD")

def _guard():
    """redirects to login if user not in session"""
    if "user" not in session:
        return redirect(url_for("login"))

# ─── Shared base HTML template ───────────────────────────────────────
BASE_HTML = """<!doctype html>
<html><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{{ 'AgriScan' }}</title>
<style>
:root{--green:#0a8400;--dark:#044a00;--light:#e8ffe8}
*{box-sizing:border-box;margin:0;padding:0;font-family:Arial}
body{display:flex;flex-direction:column;min-height:100vh;background:#f5fff5;color:#222}
nav{display:flex;justify-content:space-between;align-items:center;padding:14px 20px;background:var(--green);color:#fff}
nav .brand{font-size:1.4rem;font-weight:bold}
nav a{color:#fff;text-decoration:none;margin-left:20px}
nav .hamburger{background:none;border:none;color:#fff;font-size:1.4rem;cursor:pointer}
#sidebar{position:fixed;top:0;left:-260px;width:260px;height:100%;background:var(--dark);color:#fff;padding:80px 20px 20px;transition:.3s}
#sidebar a{display:block;color:#fff;text-decoration:none;margin:12px 0}
#sidebar.active{left:0}
main{flex:1;padding:60px 20px;text-align:center}
footer{background:#ddd;padding:12px;text-align:center;font-size:.9rem;color:#444}
.btn,button{padding:10px 22px;border:none;border-radius:6px;background:var(--green);color:#fff;cursor:pointer}
.card{display:inline-block;padding:24px;margin:14px;border-radius:12px;min-width:250px;background:#ffffffd0}
input{padding:10px;border:1px solid #999;border-radius:6px;width:220px}
</style>
<script>function toggleSidebar(){document.getElementById('sidebar').classList.toggle('active');}</script>
</head><body>
<nav>
  <button class="hamburger" onclick="toggleSidebar()">☰</button>
  <span class="brand">AgriScan</span>
  <div>
    <a href="{{ url_for('landing') }}">Home</a>
    <a href="{{ url_for('landing') }}#services">Services</a>
    <a href="{{ url_for('chat_ui') }}">ChatBot</a>
    {% if session.get('user') %}
      <a href="{{ url_for('dashboard') }}">Dashboard</a>
      <a href="{{ url_for('logout') }}">Logout</a>
    {% else %}
      <a href="{{ url_for('login') }}">Login</a>
      <a href="{{ url_for('signup') }}">Sign Up</a>
    {% endif %}
  </div>
</nav>
<div id="sidebar">
  <a href="{{ url_for('landing') }}">Home</a>
  <a href="{{ url_for('landing') }}#services">Services</a>
  <a href="{{ url_for('chat_ui') }}">ChatBot</a>
  {% if session.get('user') %}
    <a href="{{ url_for('dashboard') }}">Dashboard</a>
    <a href="{{ url_for('logout') }}">Logout</a>
  {% else %}
    <a href="{{ url_for('login') }}">Login</a>
    <a href="{{ url_for('signup') }}">Sign Up</a>
  {% endif %}
</div>
<main>{{ body|safe }}</main>
<footer>© 2025 AgriScan AI · Making farming smarter 🌱
Bringing solutions to you</footer>
</body></html>"""

def page(title, body):
    return render_template_string(BASE_HTML, title=title, body=body)

# ─── Routes ───────────────────────────────────────────────────────────
LANDING = """
<header style="padding:300px 20px;border-radius:12px;color:#fff;
background:url('https://unsplash.com/photos/green-grass-field-under-white-clouds-during-daytime-PvwdlXqo85k') center/cover;">
 <h1>Crop‑disease detection at your fingertips</h1>
 <p style="margin:18px 18px;padding-bottom:20px;font-size:1.1rem;">Snap, upload & save your harvest.<br> No need to pay a specialist</p>
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
    return page("AgriScan – Home", LANDING)

# ----------  Auth  ----------
SIGNUP = """
<h2>Create an account</h2>
<form method="post" style="margin-top:40px;">
  <input name="u" placeholder="Username"><br><br>
  <input type="password" name="p" placeholder="Password"><br><br>
  <button>Sign Up</button>
</form>
{% if error %}<p style="color:red;margin-top:14px;">{{ error }}</p>{% endif %}
<p style="margin-top:22px;">Already have an account? <a href="{{ url_for('login') }}">Log in</a></p>
"""
@app.route("/signup", methods=["GET","POST"])
def signup():
    if request.method=="POST":
        u=request.form["u"].strip().lower(); p=request.form["p"]
        if not u or not p:
            return page("Sign Up", render_template_string(SIGNUP,error="All fields required"))
        db=Session();                      # check duplicate
        if db.query(User).filter_by(username=u).first():
            db.close(); return page("Sign Up", render_template_string(SIGNUP,error="Username taken"))
        db.add(User(username=u,hash=bcrypt.hash(p))); db.commit(); db.close()
        session["user"]=u; return redirect(url_for("dashboard"))
    return page("Sign Up", render_template_string(SIGNUP,error=None))

LOGIN = """
<h2>Login</h2>
<form method="post" style="margin-top:40px;">
  <input name="u" placeholder="Username"><br><br>
  <input type="password" name="p" placeholder="Password"><br><br>
  <button>Login</button>
</form>
{% if error %}<p style="color:red;margin-top:14px;">{{ error }}</p>{% endif %}
<p style="margin-top:22px;">New here? <a href="{{ url_for('signup') }}">Create account</a></p>
"""
@app.route("/login", methods=["GET","POST"])
def login():
    if request.method=="POST":
        u=request.form["u"].strip().lower(); p=request.form["p"]
        db=Session(); user=db.query(User).filter_by(username=u).first(); db.close()
        if user and bcrypt.verify(p,user.hash):
            session["user"]=u; return redirect(url_for("dashboard"))
        return page("Login",render_template_string(LOGIN,error="Invalid credentials"))
    return page("Login",render_template_string(LOGIN,error=None))

@app.route("/logout")
def logout(): session.pop("user",None); return redirect(url_for("landing"))

# ----------  Dashboard ----------
DASH = """
<h2>Welcome, {{ user }}</h2>
<div class="card"><h3><a href="{{ url_for('scan') }}">Start Plant‑Disease Scan</a></h3></div>
<p style="margin-top:35px;"><a href="{{ url_for('logout') }}">Log out</a></p>
"""
@app.route("/dashboard")
def dashboard():
    redir=_guard();        # enforce login
    if redir: return redir
    return page("Dashboard", render_template_string(DASH,user=session["user"]))

# ----------  Scan ----------
SCAN = """
<h2>Upload Leaf Image</h2>
<form action="{{ url_for('predict') }}" method="post" enctype="multipart/form-data" style="margin-top:40px;">
  <input type="file" name="file" accept="image/*"><br><br>
  <button>Scan</button>
</form>
{% if result %}
  <h3 style="margin-top:40px;">Result</h3>
  <p>Disease/Status: <b>{{ result.class_ }}</b></p>
  <p>Confidence: {{ '{:.1%}'.format(result.confidence) }}</p>
{% endif %}
<p style="margin-top:30px;"><a href="{{ url_for('dashboard') }}">⬅ Back</a></p>
"""
@app.route("/scan")
def scan():
    redir=_guard();  # must login
    if redir: return redir
    return page("Leaf Scan", render_template_string(SCAN,result=None))

@app.route("/predict", methods=["POST"])
def predict():
    redir=_guard()
    if redir: return redir
    file=request.files.get("file")
    if not file or file.filename=="":
        return redirect(url_for("scan"))
    try:
        img=Image.open(io.BytesIO(file.read()))
        res=predict_pil(img)
        return page("Leaf Scan", render_template_string(SCAN,result=res))
    except Exception as e:
        return jsonify({"error":str(e)}),500

# ----------  ChatBot ----------
CHAT_UI = """
<h2>Ask AgriScan AI</h2>
<div id="chatBox" style="max-width:600px;margin:40px auto;border:1px solid #ccc;
     padding:16px;border-radius:8px;min-height:280px;background:#fff;overflow-y:auto;"></div>

<form id="chatForm" style="max-width:600px;margin:12px auto;" onsubmit="return sendMsg();">
  <input id="userMessage" style="width:78%;" placeholder="Type your question…">
  <button class="btn">Send</button>
</form>

<script>
async function sendMsg(){
  const inp=document.getElementById('userMessage');
  const msg=inp.value.trim();
  if(!msg) return false;
  const box=document.getElementById('chatBox');
  box.innerHTML+=`<div><b>You:</b> ${msg}</div>`;
  inp.value=""; box.scrollTop=box.scrollHeight;
  try{
    const r=await fetch("/api/chat",{method:"POST",headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:msg})});
    const d=await r.json();
    box.innerHTML+=`<div><b>Bot:</b> ${d.response || d.error}</div>`;
    box.scrollTop=box.scrollHeight;
  }catch(e){box.innerHTML+=`<div><b>Error:</b> ${e}</div>`;}
  return false;
}
</script>
"""
@app.route("/chat")
def chat_ui():
    redir=_guard();  # login required
    if redir: return redir
    return page("ChatBot", CHAT_UI)

@app.route("/api/chat", methods=["POST"])
def chat_api():
    data=request.get_json(silent=True) or {}
    prompt=data.get("message","").strip()
    if not prompt: return jsonify({"error":"Empty prompt"}),400
    try:
        resp=openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}]
        )
        return jsonify({"response":resp.choices[0].message.content})
    except Exception as e:
        return jsonify({"error":str(e)}),500

# ----------  Health ----------
@app.route("/health")
def health(): return "OK",200

# ─── Run locally ──────────────────────────────────────────────────────
if __name__=="__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT",5000)), debug=True)

  

