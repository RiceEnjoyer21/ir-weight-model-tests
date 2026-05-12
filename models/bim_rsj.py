import dataset_utils
from collections import defaultdict
import re
import numpy as np
import json
import os

class BIM_RSJ:
    def __init__(self, corpus, query, rel_ids):
        self.corpus = corpus
        self.query_terms = list(set(re.findall(r"\b\w+\b", query.lower())))
        self.index = dataset_utils.make_inverted_index(self.corpus)
        self.relevant_ids = set(rel_ids)

        self.R = len(self.relevant_ids)
        self.N = len(corpus)    
        self.NR = self.N - self.R

        self.weights = {term: self.term_weight(term) for term in self.query_terms}

    def term_weight(self, term):
        if term not in self.index:
            return 0

        docs_with_term = self.index[term]

        ri = len(docs_with_term.intersection(self.relevant_ids))
        ni = len(docs_with_term)
        nri = ni - ri

        p_rel = (ri + 0.5) / (self.R + 1)
        p_nrel = (nri + 0.5) / (self.NR + 1)
        return np.log10((p_rel*(1-p_nrel))/(p_nrel*(1-p_rel)))

    def score(self, doc):
        total = 0
        for term in self.query_terms:
            if term in doc["words"]:
                total += self.weights.get(term, 0)
        return total

    def rank(self):
        ranked_scores = []
        for doc in self.corpus:
            s = self.score(doc)
            ranked_scores.append({
                "id": doc["id"], 
                "score": round(s, 3)
            })
        return sorted(ranked_scores, key=lambda x: x["score"], reverse=True)
        
if __name__ == "__main__":
    raw_docs = dataset_utils.import_dataset('..\\data\\docs.json')

    with open('..\\data\\queries.json', 'r', encoding='utf-8') as f:
        queries_dict = json.load(f)
        
    with open('..\\data\\qrels_binary.json', 'r', encoding='utf-8') as f:
        qrels_dict = json.load(f)

    query_id = "1"
    query_text = queries_dict[query_id]

    current_qrels = qrels_dict.get(query_id, {})
    rel_ids = [d_id for d_id, val in current_qrels.items() if val == 1]

    bim = BIM_RSJ(raw_docs, query_text, rel_ids)
    results = bim.rank()

    print("-" * 50)
    print(f"Query #{query_id}: {query_text}")
    print(f"Total relevant documents in database: {len(rel_ids)}")
    print("-" * 50)
    print(f"{'Rank':<5} | {'Status':<15} | {'Doc ID':<8} | {'Score'}")
    print("-" * 50)

    for i, res in enumerate(results[:10], 1):
        doc_id = res['id']
        status = "Is Relevant" if doc_id in rel_ids else "Not Relevant"
        print(f"{i:<5} | {status:<15} | {doc_id:<8} | {res['score']}")

    ap, p10, r10, ndcg = dataset_utils.calculate_metrics(results, rel_ids, k=10)
    print("-" * 65)
    print(f"P@k: {p10:.3f} | R@k: {r10:.3f} | nDCG: {ndcg:.3f} | AP: {ap:.3f}")

