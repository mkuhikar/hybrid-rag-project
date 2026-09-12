import streamlit as st
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    """Production Cross-Encoder reranking pipeline."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model = self._load_model(model_name)

    @st.cache_resource(show_spinner=False)
    def _load_model(_self, model_name: str):
        return CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        pairs = [(query, cand["doc"]["text"]) for cand in candidates]
        scores = self.model.predict(pairs)

        for cand, score in zip(candidates, scores):
            cand["cross_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["cross_score"], reverse=True)
        return reranked[:top_k]