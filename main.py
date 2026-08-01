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

    # If the word is missing entirely from our local list, handle it gracefully
    neutral_vector = [0.0] * 50
    v1 = word_vectors.get(w1_clean, neutral_vector)
    v2 = word_vectors.get(w2_clean, neutral_vector)

    if v1 == neutral_vector or v2 == neutral_vector:
        return {"similarity": 0.00}

    # Calculate raw geometric cosine score (-1.0 to 1.0)
    raw_score = calculate_similarity(v1, v2)
    
    # FIX: Apply a contrast scaling curve to stretch out unrelated vs related words
    # This pushes cross-category words down to near 0%, and matching categories up to near 100%
    if raw_score <= 0.15:
        # Unrelated categories get dropped heavily
        percentage = max(0.0, raw_score * 20.0)
    else:
        # Matching categories get pushed into upper percentiles
        percentage = 60.0 + ((raw_score - 0.15) / (1.0 - 0.15)) * 40.0

    return {"similarity": round(percentage, 2)}
