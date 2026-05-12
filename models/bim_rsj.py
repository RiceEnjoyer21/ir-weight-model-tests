import dataset_utils
from collections import defaultdict
import re
import numpy as np
import json
import os

class BIM:
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
    
    def calculate_metrics(self, results, rel_ids, k=10):
        rel_set = set(rel_ids)
        if not rel_set or not results:
            return 0.0, 0.0, 0.0, 0.0

        ap = 0.0
        hits = 0
        dcg = 0.0

        for i, res in enumerate(results[:k], 1):
            if res['id'] in rel_set:
                hits += 1
                ap += hits/i
                dcg += 1/np.log2(i + 1)

        ap/=len(rel_set) 

        p_at_k = hits/k
        recall_at_k = hits/len(rel_set)

        idcg = sum(1 / np.log2(i+1) for i in range(1, min(len(rel_set), k) + 1))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        return ap, p_at_k, recall_at_k, ndcg
        
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

    bim = BIM(raw_docs, query_text, rel_ids)
    results = bim.rank()

    ap, p10, r10, ndcg = bim.calculate_metrics(results, rel_ids, k=10)

    print("-" * 50)
    print(f"Metrics for #{query_id}:")
    print(f"Precision@10: {p10:.3f}")
    print(f"Recall@10: {r10:.3f}")
    print(f"Avg Precision: {ap:.3f}")
    print(f"nDCG: {ndcg:.3f}")
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

