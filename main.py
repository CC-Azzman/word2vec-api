from fastapi import FastAPI, HTTPException
import json
import math
import os

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
            print(f"Loaded database vectors successfully.")
        except Exception as e:
            print(f"Error reading JSON: {e}")

def get_word_vector_or_generate(word: str):
    # If the random word exists in our engine, use it
    if word in word_vectors:
        return word_vectors[word]
        
    # FAIL-SAFE: If a random word API pulls a new word, generate a deterministic 
    # vector value using its characters so your game NEVER crashes.
    val = sum(ord(c) for c in word)
    generated_vector = [
        math.sin(val + 1) * 0.5,
        math.cos(val + 2) * 0.5,
        math.sin(val + 3) * 0.5,
        math.cos(val + 4) * 0.5,
        math.sin(val + 5) * 0.5
    ]
    return generated_vector

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

    # Fetch vectors using the generation fallback system
    v1 = get_word_vector_or_generate(w1_clean)
    v2 = get_word_vector_or_generate(w2_clean)

    score = calculate_similarity(v1, v2)
    percentage = (score + 1) / 2 * 100

    return {"similarity": round(percentage, 2)}
