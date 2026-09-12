import math
import numpy as np
from typing import List, Dict, Any

class BM25Retriever:
    """Implements BM25 sparse keyword search."""
    def __init__(self, corpus: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc["text"].split()) for doc in corpus]
        self.avgdl = sum(self.doc_len) / max(len(corpus), 1)
        self.vocab = {}
        self._index()

    def _index(self):
        for idx, doc in enumerate(self.corpus):
            words = doc["text"].lower().split()
            for word in words:
                if word not in self.vocab:
                    self.vocab[word] = {}
                self.vocab[word][idx] = self.vocab[word].get(idx, 0) + 1

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        scores = np.zeros(len(self.corpus))
        q_words = query.lower().split()
        N = len(self.corpus)

        for word in q_words:
            if word not in self.vocab:
                continue
            df = len(self.vocab[word])
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            for idx, freq in self.vocab[word].items():
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * (self.doc_len[idx] / self.avgdl))
                scores[idx] += idf * (numerator / denominator)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [{"doc_id": idx, "score": float(scores[idx]), "doc": self.corpus[idx]} for idx in top_indices if scores[idx] > 0]