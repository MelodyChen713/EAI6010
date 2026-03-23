from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from schemas import PredictRequest, PredictResponse
import model as clf

app = FastAPI(
    title="AG News Text Classification Service",
    description=(
        "A microservice for news text classification based on ULMFiT (AWD-LSTM).\n\n"
        "Classifies input news text into one of the following four categories:\n"
        "- **World**\n"
        "- **Sports**\n"
        "- **Business**\n"
        "- **Sci/Tech**"
    ),
    version="1.0.0",
)

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AG News Classifier</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f0f2f5; color: #1a1a2e; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 2rem 1rem; }
  .container { width: 100%; max-width: 720px; }
  h1 { font-size: 1.75rem; font-weight: 700; margin-bottom: .25rem; }
  .subtitle { color: #555; margin-bottom: 1.5rem; font-size: .95rem; }
  .card { background: #fff; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,.08); padding: 1.5rem; margin-bottom: 1.25rem; }
  .card h2 { font-size: 1.1rem; margin-bottom: 1rem; }
  textarea { width: 100%; min-height: 90px; padding: .75rem; border: 1px solid #d0d5dd; border-radius: 8px; font-size: .95rem; font-family: inherit; resize: vertical; transition: border .2s; }
  textarea:focus { outline: none; border-color: #4a6cf7; box-shadow: 0 0 0 3px rgba(74,108,247,.15); }
  .btn-row { display: flex; gap: .5rem; margin-top: .75rem; flex-wrap: wrap; }
  button { padding: .6rem 1.4rem; border: none; border-radius: 8px; font-size: .9rem; font-weight: 600; cursor: pointer; transition: background .2s, transform .1s; }
  button:active { transform: scale(.97); }
  .btn-primary { background: #4a6cf7; color: #fff; }
  .btn-primary:hover { background: #3b5de7; }
  .btn-primary:disabled { background: #a0b4fc; cursor: not-allowed; }
  .btn-outline { background: transparent; color: #4a6cf7; border: 1px solid #4a6cf7; }
  .btn-outline:hover { background: #f0f4ff; }
  .btn-example { background: #eef1f6; color: #333; font-size: .8rem; padding: .4rem .8rem; border-radius: 6px; }
  .btn-example:hover { background: #dde3ed; }
  #result { display: none; }
  .prediction-banner { display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap; }
  .pred-label { font-size: 1.5rem; font-weight: 700; }
  .pred-confidence { font-size: .95rem; color: #555; }
  .prob-bars { display: flex; flex-direction: column; gap: .5rem; margin-bottom: 1rem; }
  .prob-row { display: flex; align-items: center; gap: .5rem; }
  .prob-name { width: 70px; font-size: .85rem; font-weight: 600; text-align: right; }
  .prob-track { flex: 1; height: 22px; background: #eef1f6; border-radius: 6px; overflow: hidden; position: relative; }
  .prob-fill { height: 100%; border-radius: 6px; transition: width .5s ease; }
  .prob-pct { width: 50px; font-size: .85rem; color: #555; }
  .color-world { background: #f97316; }
  .color-sports { background: #22c55e; }
  .color-business { background: #3b82f6; }
  .color-scitech { background: #a855f7; }
  details { margin-top: .5rem; }
  summary { cursor: pointer; font-size: .85rem; color: #666; }
  pre { background: #f6f8fa; padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: .82rem; margin-top: .5rem; line-height: 1.5; }
  .info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .5rem .75rem; font-size: .9rem; }
  .info-grid dt { color: #888; }
  .info-grid dd { font-weight: 600; }
  .error-msg { color: #dc2626; font-weight: 600; margin-top: .5rem; }
  @media (max-width: 480px) { .info-grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="container">
  <h1>AG News Classifier</h1>
  <p class="subtitle">ULMFiT (AWD-LSTM) &mdash; classify news text into World, Sports, Business, or Sci/Tech</p>

  <div class="card">
    <h2>Predict</h2>
    <textarea id="input" placeholder="Enter news text here..."></textarea>
    <div class="btn-row">
      <button class="btn-primary" id="submitBtn" onclick="doPredict()">Classify</button>
      <button class="btn-outline" onclick="clearAll()">Clear</button>
    </div>
    <div class="btn-row" style="margin-top:.5rem">
      <span style="font-size:.8rem;color:#888;margin-right:.25rem;">Examples:</span>
      <button class="btn-example" onclick="fillExample(0)">Sci/Tech</button>
      <button class="btn-example" onclick="fillExample(1)">Sports</button>
      <button class="btn-example" onclick="fillExample(2)">Business</button>
      <button class="btn-example" onclick="fillExample(3)">World</button>
    </div>
  </div>

  <div class="card" id="result">
    <div class="prediction-banner">
      <span class="pred-label" id="predLabel"></span>
      <span class="pred-confidence" id="predConf"></span>
    </div>
    <div class="prob-bars" id="probBars"></div>
    <details>
      <summary>Show full JSON response</summary>
      <pre id="jsonBody"></pre>
    </details>
  </div>

  <div class="card">
    <h2>Service Info</h2>
    <dl class="info-grid">
      <dt>Model</dt><dd>ULMFiT (AWD-LSTM)</dd>
      <dt>Dataset</dt><dd>AG News</dd>
      <dt>Categories</dt><dd>World, Sports, Business, Sci/Tech</dd>
      <dt>API Docs</dt><dd><a href="/docs">/docs</a></dd>
    </dl>
  </div>
</div>

<script>
const examples = [
  "NASA launches new Mars exploration mission with advanced rover technology designed to search for signs of ancient life.",
  "Manchester United wins Champions League final in dramatic penalty shootout against Bayern Munich.",
  "Wall Street stocks surge as Federal Reserve signals interest rate cuts amid strong economic growth data.",
  "UN Security Council holds emergency meeting on Middle East crisis as diplomatic tensions escalate."
];
const colorMap = { "World": "color-world", "Sports": "color-sports", "Business": "color-business", "Sci/Tech": "color-scitech" };

function fillExample(i) { document.getElementById("input").value = examples[i]; }

function clearAll() {
  document.getElementById("input").value = "";
  document.getElementById("result").style.display = "none";
}

async function doPredict() {
  const text = document.getElementById("input").value.trim();
  if (!text) return;
  const btn = document.getElementById("submitBtn");
  btn.disabled = true; btn.textContent = "Classifying...";
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });
    if (!res.ok) throw new Error("Server returned " + res.status);
    const data = await res.json();
    renderResult(data);
  } catch (e) {
    document.getElementById("result").style.display = "block";
    document.getElementById("predLabel").textContent = "";
    document.getElementById("predConf").textContent = "";
    document.getElementById("probBars").innerHTML = '<p class="error-msg">Error: ' + e.message + "</p>";
    document.getElementById("jsonBody").textContent = "";
  } finally {
    btn.disabled = false; btn.textContent = "Classify";
  }
}

function renderResult(data) {
  document.getElementById("result").style.display = "block";
  document.getElementById("predLabel").textContent = data.prediction;
  document.getElementById("predConf").textContent = (data.confidence * 100).toFixed(1) + "% confidence";
  const barsHtml = Object.entries(data.probabilities)
    .sort((a, b) => b[1] - a[1])
    .map(([name, prob]) => {
      const pct = (prob * 100).toFixed(1);
      const cls = colorMap[name] || "color-world";
      return '<div class="prob-row">' +
        '<span class="prob-name">' + name + '</span>' +
        '<div class="prob-track"><div class="prob-fill ' + cls + '" style="width:' + pct + '%"></div></div>' +
        '<span class="prob-pct">' + pct + '%</span></div>';
    }).join("");
  document.getElementById("probBars").innerHTML = barsHtml;
  document.getElementById("jsonBody").textContent = JSON.stringify(data, null, 2);
}

document.getElementById("input").addEventListener("keydown", function(e) {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) doPredict();
});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def root():
    return INDEX_HTML


@app.get("/health")
def health():
    return {
        "service": "AG News Text Classification API",
        "model": "ULMFiT (AWD-LSTM)",
        "labels": clf.LABELS,
        "usage": "POST /predict with JSON body {\"text\": \"your news text here\"}",
        "docs": "/docs",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        result = clf.predict(req.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
