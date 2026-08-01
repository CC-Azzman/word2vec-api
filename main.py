from fastapi import FastAPI, HTTPException
import gensim.downloader as api

app = FastAPI()

print("Loading Word2Vec model...")
# Using a tiny 25-dimensional model so it stays within free hosting memory limits
wv = api.load('glove-wiki-gigaword-50') 
print("Model loaded successfully!")

@app.get("/similarity")
def get_similarity(w1: str, w2: str):
    if not w1 or not w2:
        raise HTTPException(status_code=400, detail="Missing words")
    try:
        # Lowercase the words since the model only recognizes lowercase
        score = float(wv.similarity(w1.lower(), w2.lower()))
        return {"similarity": round(score * 100, 2)}
    except KeyError:
        raise HTTPException(status_code=404, detail="Word not found")
