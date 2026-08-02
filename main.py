from fastapi import FastAPI, HTTPException
import json
import math
import os
import random
import hashlib

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

    # If the words are identical, return 100% immediately
    if w1_clean == w2_clean:
        return {"similarity": 100.00}

    # ========================================================
    # UNRECOGNIZED WORD CAP: 0.00% - 5.00%
    # Uses a hash function to generate fixed decimal scores in that window
    # ========================================================
    if w1_clean not in word_vectors or w2_clean not in word_vectors:
        # Create a unique integer seed out of the guessed words
        combined_string = f"{w1_clean}:{w2_clean}"
        seed_hash = int(hashlib.md5(combined_string.encode('utf-8')).hexdigest(), 16)
        
        # Pulls a consistent value strictly within the 0.00 to 5.00 range
        low_range_score = (seed_hash % 501) / 100.0
        return {"similarity": round(low_range_score, 2)}

    v1 = word_vectors[w1_clean]
    v2 = word_vectors[w2_clean]

    # Calculate raw geometric similarity
    score = calculate_similarity(v1, v2)
    raw_percentage = (score + 1) / 2 * 100.0

    if raw_percentage < 90.0:
        # Cross-category matches also map into the fixed 0.00% - 5.00% baseline window
        combined_string = f"{w1_clean}:{w2_clean}"
        seed_hash = int(hashlib.md5(combined_string.encode('utf-8')).hexdigest(), 16)
        low_range_score = (seed_hash % 501) / 100.0
        final_percentage = low_range_score
    else:
        # Matching category words stretch cleanly up into the hot zone tier
        final_percentage = 72.00 + ((raw_percentage - 90.0) / (100.0 - 90.0)) * 26.50

    return {"similarity": round(final_percentage, 2)}
