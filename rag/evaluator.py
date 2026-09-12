import re
import numpy as np
import torch
from sentence_transformers import CrossEncoder

class RAGTriadEvaluator:
    def __init__(self, embedder):
        self.embedder = embedder
        # Load DeBERTa NLI model for strict entailment/faithfulness check
        self.nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')

    def evaluate_context_relevance(self, query: str, context: str) -> float:
        """Percentage of query key tokens present in retrieved context."""
        q_words = set(re.findall(r'\w+', query.lower()))
        c_words = set(re.findall(r'\w+', context.lower()))
        if not q_words:
            return 0.0
        overlap = q_words.intersection(c_words)
        return (len(overlap) / len(q_words)) * 100.0

    def evaluate_faithfulness(self, response: str, context: str) -> float:
        """Percentage of generated statements strictly entailed by retrieved context."""
        # Split response into individual bullet points/statements
        lines = [line.strip() for line in re.split(r'[\n\.\•\-\*]+', response) if line.strip()]
        if not lines or not context.strip():
            return 0.0

        pairs = [(context, line) for line in lines]
        logits = self.nli_model.predict(pairs)
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()

        # Label 1 corresponds to 'entailment'
        entailment_scores = probs[:, 1]
        
        # Average entailment across all generated claims converted to percentage
        faithfulness_pct = np.mean(entailment_scores) * 100.0
        return float(faithfulness_pct)

    def evaluate_answer_relevance(self, query: str, response: str) -> float:
        """Cosine similarity between query and generated response."""
        q_vec = self.embedder.embed(query)
        r_vec = self.embedder.embed(response)
        
        # Cosine similarity formula
        sim = np.dot(q_vec, r_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(r_vec) + 1e-8)
        # Scale to 0% - 100%
        sim_pct = max(0.0, float(sim)) * 100.0
        return sim_pct

    def evaluate_triad(self, query: str, context: str, response: str) -> dict:
        """Returns all 3 scores as formatted percentage strings."""
        cr = self.evaluate_context_relevance(query, context)
        f = self.evaluate_faithfulness(response, context)
        ar = self.evaluate_answer_relevance(query, response)

        return {
            "context_relevance": f"{cr:.1f}%",
            "faithfulness": f"{f:.1f}%",
            "answer_relevance": f"{ar:.1f}%"
        }