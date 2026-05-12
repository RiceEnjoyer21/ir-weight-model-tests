from collections import defaultdict
import re
import json
import numpy as np

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

def calculate_metrics(results, rel_ids, k=10):
    rel_set = set(rel_ids)
    if not rel_set or not results:
        return 0.0, 0.0, 0.0, 0.0

    ap = 0.0
    hits = 0
    dcg = 0.0

    top_results = results[:k]

    for i, res in enumerate(top_results, 1):
        if res['id'] in rel_set:
            hits+=1
            ap+=hits/i
            dcg+=1/np.log2(i + 1)

    ap/=len(rel_set) 

    p_at_k = hits/k
    recall_at_k = hits/len(rel_set)

    idcg = sum(1/np.log2(i+1) for i in range(1, min(len(rel_set), k) + 1))
    ndcg = dcg/idcg if idcg > 0 else 0.0

    return ap, p_at_k, recall_at_k, ndcg