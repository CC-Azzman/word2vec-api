from fastapi import FastAPI, HTTPException
import math
import os

app = FastAPI()

# Point directly to the file inside your local folder
VECTORS_FILE = "vectors.txt"
word_vectors = {}

def load_local_vectors():
    if word_vectors:
        return
        
    print("Loading local Word2Vec vectors...")
    if not os.path.exists(VECTORS_FILE):
        print(f"Error: {VECTORS_FILE} not found in directory!")
        return

    try:
        count = 0
        with open(VECTORS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts or len(parts) < 2:
                    continue

                word = parts[0]
                try:
                    vec = [float(x) for x in parts[1:]]
                    word_vectors[word] = vec
                    count += 1
                except (ValueError, TypeError):
                    continue
                    
        print(f"Successfully loaded {count} local words into memory!")
    except Exception as e:
        print(f"Failed to read local vectors: {e}")

def calculate_similarity(v1, v2):
    dot = sum(a*b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a*a for a in v1))
    mag2 = math.sqrt(sum(a*a for a in v2))
    if not mag1 or not mag2:
        return 0.0
    return dot / (mag1 * mag2)

@app.get("/similarity")
def get_similarity(w1: str, w2: str):
    # Ensure vectors are loaded from your folder
    load_local_vectors()

    w1 = w1.lower().strip()
    w2 = w2.lower().strip()

    if w1 not in word_vectors or w2 not in word_vectors:
        raise HTTPException(
            status_code=404, 
            detail=f"Word not found. Available test words are: {', '.join(word_vectors.keys())}"
        )

    score = calculate_similarity(word_vectors[w1], word_vectors[w2])
    percentage = (score + 1) / 2 * 100

    return {"similarity": round(percentage, 2)}
