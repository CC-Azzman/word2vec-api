from fastapi import FastAPI, HTTPException
import requests
import math

app = FastAPI()

# A lightweight, clean 50-dimensional vector dataset of 10,000 common words
VECTORS_URL = "https://githubusercontent.com"
word_vectors = {}

@app.on_event("startup")
def load_vectors():
    print("Downloading lightweight Word2Vec vectors...")
    try:
        response = requests.get(VECTORS_URL, stream=True)
        count = 0
        for line in response.iter_lines():
            if line:
                parts = line.decode('utf-8').split()
                word = parts[0]
                # Limit to the top 12,000 most common words to keep memory super low
                if count > 12000:
                    break
                try:
                    word_vectors[word] = [float(x) for x in parts[1:]]
                    count += 1
                except ValueError:
                    continue
        print(f"Successfully loaded {len(word_vectors)} words into memory!")
    except Exception as e:
        print(f"Failed to load vectors: {e}")

def calculate_similarity(v1, v2):
    # Pure mathematical formula for Word2Vec similarity (Cosine Similarity)
    dot_product = sum(x * y for x, y in zip(v1, v2))
    magnitude1 = math.sqrt(sum(x * x for x in v1))
    magnitude2 = math.sqrt(sum(x * x for x in v2))
    if not magnitude1 or not magnitude2:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

@app.get("/similarity")
def get_similarity(w1: str, w2: str):
    w1_clean = w1.strip().lower()
    w2_clean = w2.strip().lower()
    
    if w1_clean not in word_vectors or w2_clean not in word_vectors:
        raise HTTPException(status_code=404, detail="Word not found")
        
    score = calculate_similarity(word_vectors[w1_clean], word_vectors[w2_clean])
    
    # Scale from -1 to 1 up to a clean 0% - 100% scale
    percentage = (score + 1) / 2 * 100
    return {"similarity": round(percentage, 2)}
