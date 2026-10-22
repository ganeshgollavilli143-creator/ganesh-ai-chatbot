"""Train the intent classifier and save its model and vocabulary files."""

import json
import pickle
import random
from pathlib import Path

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

BASE_DIR = Path(__file__).resolve().parent
lemmatizer = WordNetLemmatizer()

for package in ("punkt", "wordnet"):
    if not nltk.download(package, quiet=True):
        raise RuntimeError(f"Unable to download required NLTK resource: {package}")

with (BASE_DIR / "intents.json").open(encoding="utf-8") as data_file:
    intents = json.load(data_file)

words = []
classes = []
documents = []
ignore_words = {"?", "!"}

for intent in intents["intents"]:
    for pattern in intent.get("patterns", []):
        if not isinstance(pattern, str) or not pattern.strip():
            continue
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        documents.append((tokens, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

if not documents:
    raise ValueError("No non-empty intent patterns were found in intents.json")

words = sorted({
    lemmatizer.lemmatize(word.lower())
    for word in words
    if word not in ignore_words
})
classes = sorted(set(classes))

print(f"{len(documents)} training examples")
print(f"{len(classes)} intent classes")
print(f"{len(words)} unique words")

with (BASE_DIR / "words.pkl").open("wb") as words_file:
    pickle.dump(words, words_file)
with (BASE_DIR / "classes.pkl").open("wb") as classes_file:
    pickle.dump(classes, classes_file)

training = []
for tokens, tag in documents:
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    bag = [1 if word in pattern_words else 0 for word in words]
    output_row = [0] * len(classes)
    output_row[classes.index(tag)] = 1
    training.append((bag, output_row))

random.shuffle(training)
train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])

model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation="relu"),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(len(train_y[0]), activation="softmax"),
])
optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save(BASE_DIR / "chatbot_model.h5")
print("Model and vocabulary files saved.")
