import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

class ProductionEmbedder:
    """Singleton-style dense vector embedder using SentenceTransformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = self._load_model(model_name)

    @st.cache_resource(show_spinner=False)
    def _load_model(_self, model_name: str):
        return SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        vector = self.model.encode(text, normalize_embeddings=True)
        return np.array(vector, dtype=np.float32)