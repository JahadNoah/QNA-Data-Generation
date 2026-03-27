# QnA Pair Generator (Bahasa Melayu)

A Python web application that automatically generates high-quality Question-Answer pairs in Bahasa Melayu from `.txt` documents. Uses a six-stage AI pipeline (prefilter_v2 → factextract → generator_v2 → variation → reviewer_v2) and exports results to CSV.

LLM calls are made from the **Flask backend** to a local **Ollama** server on the LAN — the frontend never contacts the LLM directly.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running on a reachable machine with `qwen2.5:7b` pulled
- Windows / macOS / Linux

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the `.env` file

Copy `.env.example` to `.env` and set your Ollama server details:

```dotenv
# Ollama remote/local server
OLLAMA_URL=http://<SERVER_IP>:11434/api/generate
OLLAMA_MODEL=qwen2.5:7b
```

Replace `<SERVER_IP>` with the IP address of the machine running Ollama (e.g. `10.3.64.150`).  
If Ollama is running on the same machine, use `localhost`.

---

## Running the App

```bash
.venv\Scripts\python.exe web.py      # Windows
# or
python web.py                         # macOS / Linux
```

Open **http://localhost:8080** in your browser.

### Login

The app requires a login. Default credentials (hardcoded in `web.py`):

| Field    | Value                          |
|----------|--------------------------------|
| Email    | `a200363@siswa.ukm.edu.my`     |
| Password | `#LLMUKM@aiBM`                 |

To change them, edit `LOGIN_EMAIL` and `LOGIN_PASSWORD` at the top of `web.py`.

---

## How to Use

1. Log in with the credentials above.
2. Upload a `.txt` file using the file picker.
3. (Optional) Set a Q&A pair limit and select a domain.
4. Click **Jana Pasangan Q&A** to generate pairs.
5. Preview results, then click **Muat Turun CSV** to download.
6. Click **Log Keluar** in the top-right to sign out.

---

## Architecture

### LLM Call Flow

```
Browser (JS)
  → POST /api/generate  (Flask — web.py)
    → core.process_text_file()
      → core.chat()
        → requests.post("http://<OLLAMA_URL>/api/generate")
          ← { "response": "..." }
        ← plain text string
      ← list of Q&A dicts
    ← SSE stream → final pairs JSON
  ← Browser renders results
```

### Six-Step Pipeline

| Stage | Purpose |
|---|---|
| **Raw Text** | Original uploaded document text |
| **prefilter_v2** | Extracts/canonicalizes CLEAN_TEXT (TITLE, ABSTRACT, SOURCE, BODY) |
| **factextract** | Converts chunk text to atomic JSONL facts |
| **generator_v2** | Generates base Q&A by curriculum phase from facts + CLEAN_TEXT |
| **variation** | Creates controlled question variants per base Q&A |
| **reviewer_v2** | Accepts/edits/rejects candidates against chunk-grounded evidence |

### Key Files

| File | Role |
|---|---|
| `web.py` | Flask app — routes, login, SSE streaming |
| `core.py` | LLM calls, chunking, generation, review, deduplication |
| `templates/index.html` | Single-page frontend |
| `templates/login.html` | Login page |
| `prompts/` | System prompt text files for each pipeline stage |
| `.env` | Runtime configuration (Ollama URL, model name) |

---

## Optional Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | *(Cloudflare tunnel URL — see note below)* | Ollama endpoint |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Model to use |
| `QNA_CHUNK_WORDS` | `800` | Words per text chunk |
| `QNA_CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `QNA_DUP_QUESTION_SIM` | `0.88` | Fuzzy dedup threshold (0–1) |
| `QNA_MAX_PAIRS` | `100` | Hard cap on pairs per doc |

---

## Cloud Run Deployment Notes

This app is deployed on **Google Cloud Run**. The `.env` file is excluded from the Docker image (see `.dockerignore`) — environment variables must be set directly in the **Cloud Run console** under *Edit & Deploy New Revision → Variables & Secrets*.

### Key behaviour
- If no env vars are set in Cloud Run, the app falls back to the **hardcoded defaults** in `core.py`.
- If env vars **are** set in Cloud Run, they take priority over the hardcoded defaults.
- The `.env` file only applies when running the app **locally** (`python web.py`).

### Changing model or URL (no code push needed)

To switch the model or update the Cloudflare tunnel URL on the live deployment:

1. Go to [Cloud Run console](https://console.cloud.google.com/run) → select the service
2. Click **Edit & Deploy New Revision**
3. Go to **Variables & Secrets** tab
4. Update `OLLAMA_MODEL` and/or `OLLAMA_URL`
5. Click **Deploy**

> ⚠️ **Cloudflare tunnel URLs change** every time the tunnel is restarted on the Ollama server. When this happens, update `OLLAMA_URL` in the Cloud Run console — no code changes or GitHub pushes required.
