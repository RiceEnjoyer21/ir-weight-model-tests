from collections import defaultdict
import re
import numpy as np
import json
import os

corpus = [
    {"text": "python code runs fast",   "rel": True},
    {"text": "python is easy to learn", "rel": True},
    {"text": "python code is clean",    "rel": True},
    {"text": "java code runs slow",     "rel": False},
    {"text": "java is verbose",         "rel": False},
    {"text": "fast code runs well",     "rel": False},
]

query = "python code runs"

class BIM:
    def __init__(self, corpus, query):
        self.corpus = corpus
        self.query = re.findall(r"\b\w+\b", query.lower())
        self.R = sum(1 for doc in corpus if doc["rel"])
        self.NR = sum(1 for doc in corpus if not doc["rel"])

    def has_term(self, doc_text, term):
        words = re.findall(r"\b\w+\b", doc_text.lower())
        return term in words

    def _p_rel(self, term):
        ri = sum(1 for doc in self.corpus if doc["rel"] and self.has_term(doc["text"], term))
        return (ri + 0.5) / (self.R + 1), ri

    def _p_nrel(self, term):
        nri = sum(1 for doc in self.corpus if not doc["rel"] and self.has_term(doc["text"], term))        
        return (nri + 0.5) / (self.NR + 1), nri

    def term_weight(self, term):
        p_rel, ri = self._p_rel(term)
        p_nrel, nri = self._p_nrel(term)
        return np.log10((p_rel*(1-p_nrel))/(p_nrel*(1-p_rel)))
    
    def score(self, doc_text):
        total = 0
        for term in self.query:
            if self.has_term(doc_text, term):
                total += self.term_weight(term)
        return total
    
    def rank(self):
        scores = []
        for doc in self.corpus:
            s = self.score(doc["text"])
            scores.append((doc["text"], round(s, 3)))
        return sorted(scores, key=lambda x: x[1], reverse=True)
    

if __name__ == "__main__":
    bim = BIM(corpus, query)

    print("\nRanking:")
    for i, (text, score) in enumerate(bim.rank(), 1):
        print(f"{i}.[{score}] {text}")