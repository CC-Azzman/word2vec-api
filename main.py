from fastapi import FastAPI, HTTPException
import requests
import math

app = FastAPI()

# MUST be a real raw text file containing word vectors
VECTORS_URL = "https://githubusercontent.com"

word_vectors = {}

@app.on_event("startup")
def load_vectors():
    print("Downloading lightweight Word2Vec vectors...")
    try:
        response = requests.get(VECTORS_URL)
        response.raise_for_status()

        count = 0
        for line in response.text.splitlines():
            parts = line.strip().split()
            if not parts:
                continue

            word = parts[0]

            if count >= 12000:
                break

            try:
                vec = [float(x) for x in parts[1:]]
                word_vectors[word] = vec
                count += 1
            except ValueError:
                continue

        print(f"Loaded {len(word_vectors)} words.")
    except Exception as e:
        print(f"Failed to load vectors: {e}")

def calculate_similarity(v1, v2):
    dot = sum(a*b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a*a for a in v1))
    mag2 = math.sqrt(sum(a*a for a in v2))
    if not mag1 or not mag2:
        return 0.0
    return dot / (mag1 * mag2)

@app.get("/similarity")
def get_similarity(w1: str, w2: str):
    w1 = w1.lower().strip()
    w2 = w2.lower().strip()

    if w1 not in word_vectors or w2 not in word_vectors:
        raise HTTPException(status_code=404, detail="Word not found")

    score = calculate_similarity(word_vectors[w1], word_vectors[w2])

    # Semantle-style percentage
    percentage = max(0.0, round(score * 100, 2))

    return {"similarity": percentage}
