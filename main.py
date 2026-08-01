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
            print("Loaded database vectors successfully.")
        except Exception as e:
            print(f"Error reading JSON: {e}")

@app.get("/random-word")
def get_random_word():
    if not word_vectors:
        raise HTTPException(status_code=500, detail="Database empty")
    random_secret = random.choice(list(word_vectors.keys()))
    return {"word": random_secret}

def get_word_vector_or_generate(word: str):
    if word in word_vectors:
        return word_vectors[word]
        
    # FIX: Generates exactly 50 dimensions to perfectly match vectors.json
    val = sum(ord(c) for c in word)
    generated_vector = []
    for i in range(1, 51):
        if i % 2 == 0:
            generated_vector.append(math.cos(val + i) * 0.5)
        else:
            generated_vector.append(math.sin(val + i) * 0.5)
            
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

    v1 = get_word_vector_or_generate(w1_clean)
    v2 = get_word_vector_or_generate(w2_clean)

    score = calculate_similarity(v1, v2)
    percentage = (score + 1) / 2 * 100

    return {"similarity": round(percentage, 2)}
