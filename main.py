from fastapi import FastAPI, HTTPException
import json
import math
import os
import random

app = FastAPI()

VECTORS_FILE = "vectors.json"
word_vectors = {}

@app.on_event("startup")
def load_local_vectors():
    global word_vectors
    print("Loading local JSON database...")
    if os.path.exists(VECTORS_FILE):
        try:
            with open(VECTORS_FILE, "r", encoding="utf-8") as f:
                word_vectors = json.load(f)
            print(f"Loaded {len(word_vectors)} words successfully.")
        except Exception as e:
            print(f"Error reading JSON: {e}")

@app.get("/random-word")
def get_random_word():
    if not word_vectors:
        raise HTTPException(status_code=500, detail="Database empty")
    random_secret = random.choice(list(word_vectors.keys()))
    return {"word": random_secret}

def calculate_similarity(v1, v2):
    dot = sum(a*b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a*a for a in v1))
    mag2 = math.sqrt(sum(a*a for a in v2))
    if not mag1 or not mag2:
        return 0.0
    return dot / (mag1 * mag2)

@app.get("/similarity")
def get_similarity(w1: str, w2: str):
    w1_clean = w1.lower().strip()
    w2_clean = w2.lower().strip()

    if not w1_clean or not w2_clean:
        raise HTTPException(status_code=400, detail="Missing words")

    # Safe fallback if a word isn't recognized
    if w1_clean not in word_vectors or w2_clean not in word_vectors:
        return {"similarity": 5.00}

    # If the words are identical, return 100.00 immediately
    if w1_clean == w2_clean:
        return {"similarity": 100.00}

    score = calculate_similarity(word_vectors[w1_clean], word_vectors[w2_clean])
    
    # Scale mathematical cosine values (-1.0 to 1.0) onto a standard 0-100 baseline
    raw_percentage = (score + 1) / 2 * 100.0

    # ========================================================
    # GAME INFLECTION CALIBRATION CURVE
    # Chops off the ~59% cross-category baseline and stretches it
    # ========================================================
    baseline = 59.5
    if raw_percentage <= baseline:
        # Scale unrelated words dynamically between 0.00% and 8.00%
        final_percentage = max(0.00, (raw_percentage / baseline) * 8.00)
    else:
        # Stretch matching categories beautifully between 40.00% and 99.50%
        final_percentage = 40.00 + ((raw_percentage - baseline) / (100.0 - baseline)) * 59.50

    return {"similarity": round(final_percentage, 2)}
