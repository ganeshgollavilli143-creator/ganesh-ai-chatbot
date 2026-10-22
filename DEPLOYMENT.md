# Running and deploying Ganesh AI Chatbot

## Run locally

Use Python 3.9–3.11. Create and activate a virtual environment, install the packages, and start the app:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open <http://localhost:8000>. NLTK downloads its tokenizer and WordNet data on first start when needed.

## Docker

Build and run from the project directory:

```bash
docker build -t ganesh-ai-chatbot .
docker run --rm -p 8000:8000 ganesh-ai-chatbot
```

Then visit <http://localhost:8000>. Docker Compose is also provided; `docker compose up --build` starts the app and its optional Nginx proxy.

## Hosting

A host must support Python 3.11, the pinned TensorFlow dependency, and a persistent copy of the model files. Configure the start command as:

```text
uvicorn app:app --host 0.0.0.0 --port $PORT
```

Check the selected hosting provider's current Python and memory limits before deploying. The project is intended as a learning/demo app and does not include production authentication or monitoring.
