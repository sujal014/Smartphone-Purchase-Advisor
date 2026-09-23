# 📱 Smartphone Purchase Advisor

A mini project combining **LangChain (LLM reasoning)** and a genuine **Fuzzy
Inference System (scikit-fuzzy)** behind a **Streamlit** UI.

- **LangChain / LLM** (via Groq's free API, running Llama 3.3 70B) reads a
  free-text request ("I need a phone under 30k, I take a lot of photos,
  battery dies fast on my current phone, don't game much") and extracts a
  budget + 1-10 importance weights for camera, performance, and battery —
  then later explains the results conversationally.
- **Fuzzy logic (Mamdani FIS)**: real membership functions (`low` / `medium`
  / `high`), fuzzification, a 10-rule rule base, and centroid defuzzification
  — not if/else — score every phone in the catalog for how well it fits the
  user's weighted priorities and budget.

## File structure

```
smartphone-advisor/
├── app.py                          # Streamlit UI — wires everything together
├── fuzzy_engine.py                 # The fuzzy inference system (skfuzzy)
├── llm_agent.py                    # LangChain: extraction + explanation
├── phones_data.py                  # Small hardcoded phone catalog
├── requirements.txt                # Pinned dependencies
├── .gitignore
└── .streamlit/
    └── secrets.toml.example        # Template — copy to secrets.toml locally
```

## 1. Run it locally

```bash
# 1. Clone your repo (after you push it — see step 3) or work in this folder
cd smartphone-advisor

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key (free — get one at https://console.groq.com/keys)
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml and paste your real key

# 5. Run the app
streamlit run app.py
```

It will open at `http://localhost:8501`.

> **Note:** `app.py` reads the key from either the `GROQ_API_KEY`
> environment variable or `st.secrets`, so `secrets.toml` is the easiest
> path locally, and Streamlit Cloud's Secrets manager covers deployment.

## 2. Test the fuzzy engine on its own (optional, no API key needed)

```bash
python3 -c "
from fuzzy_engine import build_fuzzy_system, score_phone
from phones_data import PHONES
system = build_fuzzy_system()
for p in PHONES:
    r = score_phone(system, p['price'], 30000, p['camera'], p['performance'], p['battery'],
                     camera_importance=9, performance_importance=4, battery_importance=8)
    print(p['name'], r['suitability'])
"
```

## 3. Push to GitHub

```bash
git init
git add .
git commit -m "Smartphone Purchase Advisor: LangChain + Fuzzy Logic + Streamlit"
git branch -M main
git remote add origin https://github.com/<your-username>/smartphone-advisor.git
git push -u origin main
```

`.streamlit/secrets.toml` is in `.gitignore`, so your real API key never gets
committed — only `secrets.toml.example` does.

## 4. Deploy for free (Streamlit Community Cloud)

1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. Click **"New app"** → pick your `smartphone-advisor` repo, branch `main`,
   main file path `app.py`.
3. Before/after deploying, open **Settings → Secrets** on the app and paste:
   ```toml
   GROQ_API_KEY = "gsk_...your-real-key..."
   ```
4. Click **Deploy**. You'll get a live URL like
   `https://your-app-name.streamlit.app`.

### Alternative: Hugging Face Spaces

1. Create a new Space at **https://huggingface.co/new-space**, SDK =
   **Streamlit**.
2. Push this same repo to the Space's git remote (HF Spaces are git repos
   too), or upload the files via the web UI.
3. In the Space's **Settings → Repository secrets**, add
   `GROQ_API_KEY`.
4. The Space auto-builds `app.py` — no extra config needed beyond
   `requirements.txt`, which is already here.

## How the fuzzy logic actually works (for your write-up / demo)

`fuzzy_engine.py` builds a Mamdani-style system with `skfuzzy.control`:

- **Antecedents (inputs):** `price_fit` (0-100), `camera` (0-10),
  `performance` (0-10), `battery` (0-10) — each with triangular membership
  functions for `low`, `medium`, `high`.
- **Consequent (output):** `suitability` (0-100), also `low` / `medium` /
  `high`.
- **Rule base:** 10 rules such as
  `IF price_fit is high AND camera is high AND performance is high THEN suitability is high`
  and `IF price_fit is low THEN suitability is low`.
- **Fuzzification → rule firing (min/max) → aggregation → centroid
  defuzzification** all happen inside `ControlSystemSimulation.compute()`,
  producing one crisp score per phone, which the app then ranks.

Each phone's `camera` / `performance` / `battery` inputs are pre-weighted by
the user's stated importance (extracted by the LLM) before being fed into
the fuzzy system — so a great camera phone scores high only if the user
actually said camera matters to them.

## Swapping in a different LLM provider

`llm_agent.py` uses `langchain_groq.ChatGroq` with the `llama-3.3-70b-versatile`
model (free tier, generous rate limits). To use a different provider (e.g.
OpenAI), install its LangChain package, replace the import/model in
`_get_llm()`, and swap the secret name back to whatever that provider expects.
The extraction/explanation logic itself doesn't need to change.
