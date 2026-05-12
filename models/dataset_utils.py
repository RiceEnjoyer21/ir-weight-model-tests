from collections import defaultdict
import re
import json

def import_dataset(docs_path):
    processed_docs = []
    with open(docs_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for doc_id, text in data.items():
        text = text.lower()
        text = text.replace('\n', ' ')
        text = re.sub(r"[^a-z0-9\s]+", '', text)
        text = " ".join(text.split())
        processed_docs.append({
            "id": doc_id,
            "text": text,
            "words": set(text.split())
        })

    return processed_docs

def make_inverted_index(corpus):
    index = defaultdict(set)
    for doc in corpus:
        id = doc["id"]
        words = doc["text"].split()
        for term in words:
            index[term].add(id)
    return index