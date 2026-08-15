
import os
import joblib
from flask import Flask, request, jsonify, render_template_string


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "spam_model.joblib")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\nfile doesn't exist\n"
    )

bundle = joblib.load(MODEL_PATH)
tfidf = bundle["tfidf"]
model = bundle["model"]
TEST_ACCURACY = bundle["accuracy"]

print(f"[spam-checker] model loaded — held-out test accuracy: {TEST_ACCURACY}%")

# ------------------------------------------------------------------
# 2. Flask app
# ------------------------------------------------------------------
app = Flask(__name__)

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Spam Checker — ML Classifier</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;700;900&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#0a0a0b;
    --border:rgba(255,255,255,.12);
    --border-strong:rgba(255,255,255,.28);
    --text:#f2f2f0;
    --muted:#888d94;
    --accent:#5b8cff;
    --spam:#ff4d5e;
    --ham:#34d399;
  }
  *{box-sizing:border-box;}
  html,body{margin:0;padding:0;}
  body{
    background:var(--bg);
    color:var(--text);
    font-family:'Inter',sans-serif;
    min-height:100vh;
    position:relative;
    overflow-x:hidden;
  }
  .grid-overlay{
    position:fixed;inset:0;
    background-image:
      linear-gradient(rgba(255,255,255,.045) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255,255,255,.045) 1px, transparent 1px);
    background-size:56px 56px;
    pointer-events:none;
    z-index:0;
    mask-image:radial-gradient(circle at 20% 10%, black 0%, transparent 70%);
  }
  header{
    display:flex;justify-content:space-between;align-items:center;
    padding:24px 48px;border-bottom:1px solid var(--border);
    position:relative;z-index:2;
  }
  .logo{display:flex;align-items:center;gap:12px;}
  .logo-box{
    width:38px;height:38px;border:1px solid var(--border-strong);
    display:flex;align-items:center;justify-content:center;
    font-family:'JetBrains Mono',monospace;font-weight:700;font-size:13px;
  }
  .logo-text{
    font-family:'JetBrains Mono',monospace;font-size:13px;
    letter-spacing:3px;color:var(--muted);
  }
  .acc-badge{
    font-family:'JetBrains Mono',monospace;font-size:11px;
    letter-spacing:2px;color:var(--muted);
    border:1px solid var(--border);border-radius:4px;
    padding:8px 14px;
  }
  .acc-badge span{color:var(--ham);font-weight:700;}
  main{
    max-width:880px;margin:0 auto;
    padding:72px 24px 120px;
    position:relative;z-index:2;
  }
  .meta-line{
    font-family:'JetBrains Mono',monospace;font-size:12px;
    letter-spacing:2px;color:var(--muted);text-transform:uppercase;
    display:flex;align-items:center;gap:14px;flex-wrap:wrap;
    margin-bottom:48px;
  }
  .dot{
    width:6px;height:6px;border-radius:50%;background:var(--accent);
    box-shadow:0 0 8px 2px var(--accent);display:inline-block;
  }
  .sep{color:var(--border-strong);}
  h1.hero{
    font-family:'Inter',sans-serif;font-weight:900;
    line-height:.95;letter-spacing:-2px;
    font-size:clamp(48px,9vw,104px);
    margin:0 0 32px;
  }
  h1.hero .fill{display:block;color:var(--text);}
  h1.hero .outline{
    display:block;color:transparent;
    -webkit-text-stroke:2px var(--text);
  }
  .tagline{
    font-family:'JetBrains Mono',monospace;font-size:14px;
    letter-spacing:3px;color:var(--muted);text-transform:uppercase;
    margin:0 0 18px;
  }
  .desc{
    max-width:540px;color:#b6b9bf;line-height:1.65;
    font-size:16px;margin:0 0 56px;
  }
  .panel{
    border:1px solid var(--border);border-radius:10px;
    padding:32px;background:rgba(255,255,255,.015);
  }
  textarea{
    width:100%;min-height:150px;background:transparent;
    border:1px solid var(--border);border-radius:6px;
    color:var(--text);padding:16px;font-family:'Inter',sans-serif;
    font-size:15px;line-height:1.5;resize:vertical;outline:none;
    transition:border-color .2s;
  }
  textarea:focus{border-color:var(--accent);}
  textarea::placeholder{color:#5c6067;}
  .row{
    display:flex;align-items:center;justify-content:space-between;
    margin-top:20px;flex-wrap:wrap;gap:12px;
  }
  .hint{
    font-family:'JetBrains Mono',monospace;font-size:11px;
    letter-spacing:1px;color:#5c6067;text-transform:uppercase;
  }
  button#checkBtn{
    font-family:'JetBrains Mono',monospace;text-transform:uppercase;
    letter-spacing:2px;background:var(--text);color:#0a0a0b;
    border:none;padding:14px 34px;border-radius:6px;
    font-weight:700;font-size:13px;cursor:pointer;
    transition:transform .15s ease, opacity .15s ease;
  }
  button#checkBtn:hover{transform:translateY(-2px);}
  button#checkBtn:disabled{opacity:.5;cursor:not-allowed;transform:none;}
  .result{
    display:none;margin-top:32px;padding-top:32px;
    border-top:1px solid var(--border);
  }
  .result.show{display:block;}
  .result-label{
    font-family:'Inter',sans-serif;font-weight:900;
    font-size:clamp(32px,6vw,56px);letter-spacing:-1px;
    margin:0 0 24px;transition:color .3s;
  }
  .result-label.spam{color:var(--spam);text-shadow:0 0 30px rgba(255,77,94,.35);}
  .result-label.ham{color:var(--ham);text-shadow:0 0 30px rgba(52,211,153,.35);}
  .confidence-row{
    display:flex;align-items:center;gap:16px;
    font-family:'JetBrains Mono',monospace;font-size:11px;
    letter-spacing:2px;color:var(--muted);text-transform:uppercase;
  }
  .bar{
    flex:1;height:6px;background:rgba(255,255,255,.08);
    border-radius:3px;overflow:hidden;
  }
  .bar-fill{
    height:100%;width:0%;background:var(--accent);
    transition:width .7s cubic-bezier(.2,.8,.2,1), background .3s;
    border-radius:3px;
  }
  .bar-fill.spam{background:var(--spam);}
  .bar-fill.ham{background:var(--ham);}
  #confValue{color:var(--text);font-weight:700;min-width:52px;text-align:right;}
  footer{
    font-family:'JetBrains Mono',monospace;font-size:11px;
    letter-spacing:1px;color:#5c6067;text-transform:uppercase;
    margin-top:24px;
  }
</style>
</head>
<body>
<div class="grid-overlay"></div>

<header>
  <div class="logo">
    <div class="logo-box">AI</div>
    <div class="logo-text">SPAM CHECKER</div>
  </div>
  <div class="acc-badge">TEST ACCURACY&nbsp;&nbsp;<span>{{ accuracy }}%</span></div>
</header>

<main>
  <div class="meta-line">
    EMAIL CLASSIFIER &copy; 2026
    <span class="sep">/</span>
    <span class="dot"></span> MODEL READY
    <span class="sep">/</span>
    TF-IDF + NAIVE BAYES
  </div>

  <h1 class="hero">
    <span class="fill">IS IT</span>
    <span class="outline">SPAM?</span>
  </h1>

  <p class="tagline">Machine learning spam detection</p>
  <p class="desc">
    Paste an email or message below. A TF-IDF vectorizer feeds a
    Multinomial Naive Bayes model trained on real spam/ham data,
    and it tells you which one it is — with a confidence score.
  </p>

  <div class="panel">
    <textarea id="messageInput" placeholder="Paste your email or message here..."></textarea>
    <div class="row">
      <span class="hint">Ctrl + Enter to check</span>
      <button id="checkBtn">CHECK MESSAGE</button>
    </div>

    <div class="result" id="result">
      <div class="result-label" id="resultLabel">—</div>
      <div class="confidence-row">
        <span>CONFIDENCE</span>
        <div class="bar"><div class="bar-fill" id="barFill"></div></div>
        <span id="confValue">0%</span>
      </div>
    </div>
  </div>

  <footer>Model: TF-IDF (3000 features) &middot; Multinomial Naive Bayes &middot; held-out test accuracy {{ accuracy }}%</footer>
</main>

<script>
  const btn = document.getElementById('checkBtn');
  const input = document.getElementById('messageInput');
  const result = document.getElementById('result');
  const label = document.getElementById('resultLabel');
  const barFill = document.getElementById('barFill');
  const confValue = document.getElementById('confValue');

  async function checkMessage(){
    const message = input.value.trim();
    if(!message){ input.focus(); return; }

    btn.disabled = true;
    btn.textContent = 'CHECKING...';

    try{
      const res = await fetch('/predict', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({message})
      });
      const data = await res.json();

      if(data.error){
        label.className = 'result-label';
        label.textContent = data.error;
        result.classList.add('show');
        barFill.style.width = '0%';
        confValue.textContent = '';
        return;
      }

      const isSpam = data.label === 'spam';
      label.textContent = isSpam ? 'THIS IS SPAM' : 'THIS IS HAM (SAFE)';
      label.className = 'result-label ' + (isSpam ? 'spam' : 'ham');
      barFill.className = 'bar-fill ' + (isSpam ? 'spam' : 'ham');

      result.classList.add('show');
      barFill.style.width = '0%';
      requestAnimationFrame(() => { barFill.style.width = data.confidence + '%'; });
      confValue.textContent = data.confidence + '%';
    } catch(err){
      label.className = 'result-label';
      label.textContent = 'Connection error — is the server running?';
      result.classList.add('show');
    } finally {
      btn.disabled = false;
      btn.textContent = 'CHECK MESSAGE';
    }
  }

  btn.addEventListener('click', checkMessage);
  input.addEventListener('keydown', (e) => {
    if(e.key === 'Enter' && e.ctrlKey) checkMessage();
  });
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML_PAGE, accuracy=TEST_ACCURACY)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "Message is empty"}), 400

    vec = tfidf.transform([message])
    prediction = int(model.predict(vec)[0])
    proba = model.predict_proba(vec)[0]
    confidence = round(float(proba[prediction]) * 100, 2)
    label = "spam" if prediction == 1 else "ham"

    return jsonify({"label": label, "confidence": confidence})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
