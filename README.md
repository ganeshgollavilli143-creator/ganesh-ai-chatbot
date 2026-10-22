# Ganesh AI Chatbot

A small intent-based chatbot with a FastAPI backend and a browser chat interface. It uses a TensorFlow model and NLTK to match messages to responses in `intents.json`.

> This project is adapted from the MIT-licensed source credited in [LICENSE](LICENSE). The original license notice is preserved; this version is customized and maintained by Ganesh.

## Run locally

Requirements: Python 3.9–3.11 and pip. TensorFlow 2.14 does not support every Python version, so use one in this range.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

### macOS or Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

The app downloads the required NLTK data if it is missing. Open <http://localhost:8000> in your browser. The interactive API docs are at <http://localhost:8000/docs>.

## Project files

- `app.py` — FastAPI server and chat API.
- `templates/index.html` and `static/` — browser interface and styles.
- `intents.json` — example phrases and responses used by the chatbot.
- `chatbot_model.h5`, `words.pkl`, and `classes.pkl` — trained model and vocabulary data used by the app.
- `train.py` — rebuilds the model and vocabulary from `intents.json`.

To retrain, activate the virtual environment and run `python train.py`. This overwrites the model and vocabulary files.

## API

`POST /api/chat` accepts JSON such as `{"msg":"hello"}` and returns a response and confidence score. `GET /health` reports whether the server is running.

## Notes

This is a learning/demo project. Its responses are limited to the examples in `intents.json`; it is not a general-purpose AI assistant. The web page loads Bootstrap, Font Awesome, and jQuery from public CDNs, so those visual assets need an internet connection.
