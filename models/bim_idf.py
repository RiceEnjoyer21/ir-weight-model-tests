import dataset_utils
from collections import defaultdict
import re
import numpy as np
import json
import os

class BIM_IDF:
    def __init__(self, corpus, query):
        self.corpus = corpus
        self.query_terms = list(set(re.findall(r"\b\w+\b", query.lower())))
        self.index = dataset_utils.make_inverted_index(self.corpus)

        self.N = len(corpus)

        self.weights = {term: self.term_weight(term) for term in self.query_terms}
        
    def term_weight(self, term):
        if term not in self.index:
            return 0
        
        ni = len(self.index[term])

        numerator = self.N - ni + 0.5
        denominator = ni + 0.5
        return np.log10(numerator/denominator)
    
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
        rel_ids = [d_id for d_id, val in qrels_dict.get(query_id, {}).items() if val == 1]

        bim = BIM_IDF(raw_docs, query_text)
        results = bim.rank()

        print("-" * 50)
        print(f"Results for Query #{query_id}:")
        print(f"Text: {query_text}")
        print("-" * 65)
        print(f"{'Rank':<5} | {'Status':<15} | {'Doc ID':<8} | {'Score'}")
        print("-" * 65)

        for i, res in enumerate(results[:10], 1):
            status = "Is Relevant" if res['id'] in rel_ids else "Not Relevant"
            print(f"{i:<5} | {status:<15} | {res['id']:<8} | {res['score']}")

        ap, p10, r10, ndcg = dataset_utils.calculate_metrics(results, rel_ids, k=10)
        print("-" * 65)
        print(f"P@k: {p10:.3f} | R@k: {r10:.3f} | nDCG: {ndcg:.3f} | AP: {ap:.3f}")