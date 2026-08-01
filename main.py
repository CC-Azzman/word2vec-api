from fastapi import FastAPI, HTTPException
import requests
import math

app = FastAPI()

# A verified, live public URL containing thousands of pre-made word vectors
VECTORS_URL = "https://githubusercontent.com"

word_vectors = {}

def load_vectors_if_empty():
    # If we already have the words, skip downloading
    if word_vectors:
        return
        
    print("Downloading lightweight Word2Vec vectors...")
    try:
        response = requests.get(VECTORS_URL, timeout=30)
        response.raise_for_status()

        count = 0
        for line in response.text.splitlines():
            parts = line.strip().split()
            if not parts or len(parts) < 2:
                continue

            word = parts[0]

            # Limit to the top 12,000 most common words to keep memory super low on Render
            if count >= 12000:
                break

            try:
                vec = [float(x) for x in parts[1:]]
                word_vectors[word] = vec
                count += 1
            except (ValueError, TypeError):
                continue

        print(f"Successfully loaded {len(word_vectors)} words.")
    except Exception as e:
        print(f"Failed to load vectors: {e}")
        raise HTTPException(status_code=500, detail="Word dictionary failed to load on the server. Try again in a moment.")

def calculate_similarity(v1, v2):
    dot = sum(a*b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a*a for a in v1))
    mag2 = math.sqrt(sum(a*a for a in v2))
    if not mag1 or not mag2:
        return 0.0
    return dot / (mag1 * mag2)

@app.get("/similarity")
def get_similarity(w1: str, w2: str):
    # Ensure vectors are loaded safely when a request comes in
    load_vectors_if_empty()

    w1 = w1.lower().strip()
    w2 = w2.lower().strip()

    if w1 not in word_vectors or w2 not in word_vectors:
        raise HTTPException(status_code=404, detail="Word not found")

    score = calculate_similarity(word_vectors[w1], word_vectors[w2])

    # Turn the score into a clean 0% - 100% scale
    percentage = (score + 1) / 2 * 100

    return {"similarity": round(percentage, 2)}
