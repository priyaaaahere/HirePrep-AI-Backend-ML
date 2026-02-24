import json
import re
from gensim.models import Word2Vec

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()

with open("jobs.json", "r", encoding="utf-8") as f:
    jobs = json.load(f)

sentences = []
for job in jobs:
    sentences.append(tokenize(job["job_title"]))

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)

model.save("models/word2vec.model")
print("Word2Vec model trained and saved.")