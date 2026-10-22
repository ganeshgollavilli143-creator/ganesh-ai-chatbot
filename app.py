"""FastAPI server for Ganesh AI Chatbot."""

import json
import logging
import os
import pickle
import random
from pathlib import Path

import nltk
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from nltk.stem import WordNetLemmatizer
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)
lemmatizer = WordNetLemmatizer()


def ensure_nltk_data():
    """Download tokenizer resources when they are missing from this machine."""
    resources = (("tokenizers/punkt", "punkt"), ("corpora/wordnet", "wordnet"))
    for resource, package in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            if not nltk.download(package, quiet=True):
                raise RuntimeError(f"Unable to download required NLTK resource: {package}")


ensure_nltk_data()

app = FastAPI(
    title="Ganesh AI Chatbot API",
    description="An intent-based chatbot demo built with FastAPI and TensorFlow.",
    version="1.0.0",
)

# The bundled interface uses the same origin. Add specific origins only when
# connecting a separate frontend, via CORS_ORIGINS=https://example.com.
cors_origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "").split(",") if origin.strip()]
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

try:
    model = load_model(BASE_DIR / "chatbot_model.h5")
    with (BASE_DIR / "words.pkl").open("rb") as words_file:
        words = pickle.load(words_file)
    with (BASE_DIR / "classes.pkl").open("rb") as classes_file:
        classes = pickle.load(classes_file)
    with (BASE_DIR / "intents.json").open(encoding="utf-8") as intents_file:
        intents = json.load(intents_file)
except Exception as exc:
    raise RuntimeError(f"Could not load the chatbot model or data files: {exc}") from exc


class MessageRequest(BaseModel):
    msg: str = Field(min_length=1, max_length=1000)


class ChatResponse(BaseModel):
    response: str
    confidence: float


@app.get("/", response_class=HTMLResponse)
async def home():
    page = (BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=page)


@app.post("/api/chat", response_model=ChatResponse)
async def chatbot_response(request: MessageRequest):
    msg = request.msg.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        predicted_intents = predict_class(msg, model)
        confidence = float(predicted_intents[0]["probability"]) if predicted_intents else 0.0
        response = getResponse(predicted_intents, intents)

        lower_msg = msg.lower()
        for prefix in ("hi my name is", "my name is", "i am", "i'm"):
            if lower_msg.startswith(prefix):
                name = msg[len(prefix):].strip()
                if name:
                    response = response.replace("{n}", name)
                break

        return ChatResponse(response=response, confidence=confidence)
    except Exception:
        logger.exception("Failed to process chatbot message")
        raise HTTPException(status_code=500, detail="Sorry, the chatbot could not process that message.") from None


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "loaded"}


def clean_up_sentence(sentence):
    """Tokenize and lemmatize a sentence."""
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]


def bow(sentence, vocabulary, show_details=False):
    """Create a bag-of-words array for a sentence."""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(vocabulary)
    for sentence_word in sentence_words:
        for index, vocabulary_word in enumerate(vocabulary):
            if vocabulary_word == sentence_word:
                bag[index] = 1
                if show_details:
                    logger.debug("Found in bag: %s", vocabulary_word)
    return np.array(bag)


def predict_class(sentence, chatbot_model):
    """Predict the intent class from a sentence."""
    bag = bow(sentence, words)
    predictions = chatbot_model.predict(np.array([bag]), verbose=0)[0]
    error_threshold = 0.25
    results = [(index, score) for index, score in enumerate(predictions) if score > error_threshold]
    results.sort(key=lambda result: result[1], reverse=True)
    return [
        {"intent": classes[index], "probability": str(score)}
        for index, score in results
    ]


def getResponse(predicted_intents, intents_json):
    """Choose a response for the predicted intent, or a fallback if none matched."""
    if not predicted_intents:
        return "Sorry, I didn't understand that. Could you please rephrase?"

    tag = predicted_intents[0]["intent"]
    for intent in intents_json["intents"]:
        if intent["tag"] == tag and intent.get("responses"):
            return random.choice(intent["responses"])
    return "Sorry, I didn't understand that. Could you please rephrase?"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
