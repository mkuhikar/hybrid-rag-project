"""Retrieval, reranking, answer generation, and quality evaluation."""

import logging
import os
from typing import Any

from groq import Groq
import numpy as np
from sentence_transformers import CrossEncoder

from db import DynamoVectorStore, ProductionEmbedder
from services.storage import S3Storage

LOGGER = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        LOGGER.info("Loading reranker model", extra={"model_name": model_name})
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int = 10) -> list[dict[str, Any]]:
        if not candidates:
            return []
        scores = self.model.predict([(query, candidate["text"]) for candidate in candidates])
        for candidate, score in zip(candidates, scores):
            candidate["cross_score"] = float(score)
        return sorted(candidates, key=lambda candidate: candidate["cross_score"], reverse=True)[:top_k]


class RAGTriadEvaluator:
    def __init__(self, embedder: ProductionEmbedder):
        self.embedder = embedder

    def context_relevance(self, query: str, context: str) -> float:
        query_words = set(query.lower().split())
        return float(len(query_words & set(context.lower().split())) / len(query_words)) if query_words else 0.0

    def faithfulness(self, answer: str, context: str) -> float:
        sentences = [sentence.strip() for sentence in answer.split(".") if sentence.strip()]
        context_words = set(context.lower().split())
        grounded = [len(set(sentence.lower().split()) & context_words) / max(len(set(sentence.lower().split())), 1) >= 0.5 for sentence in sentences]
        return float(sum(grounded) / len(sentences)) if sentences else 0.0

    def answer_relevance(self, query: str, answer: str) -> float:
        return float(np.dot(self.embedder.embed(query), self.embedder.embed(answer)))


class HybridRAGPipeline:
    def __init__(self, vector_store: DynamoVectorStore, storage: S3Storage):
        self.vector_store = vector_store
        self.storage = storage
        self.llm_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.reranker = CrossEncoderReranker()
        self.evaluator = RAGTriadEvaluator(vector_store.embedder)

    def query(self, question: str, top_k_parents: int = 2) -> dict[str, Any]:
        candidates = self.vector_store.search_all_sources(question)
        reranked = self.reranker.rerank(question, candidates)
        parent_ids: list[str] = []
        for item in reranked:
            if item["parent_id"] not in parent_ids:
                parent_ids.append(item["parent_id"])
            if len(parent_ids) == top_k_parents:
                break
        context = "\n---\n".join(
            self.storage.read_text(self.vector_store.get_parent_metadata(parent_id)["object_key"])
            for parent_id in parent_ids
        )
        completion = self.llm_client.chat.completions.create(model="openai/gpt-oss-20b", temperature=0.0, messages=[{"role": "system", "content": "Answer only from the supplied context. Say when the context is insufficient."}, {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}])
        answer = completion.choices[0].message.content or ""
        return {"generated_answer": answer, "matched_candidates": [item["text"] for item in reranked[:top_k_parents]], "rag_triad_scores": {"context_relevance": round(self.evaluator.context_relevance(question, context), 4), "faithfulness": round(self.evaluator.faithfulness(answer, context), 4), "answer_relevance": round(self.evaluator.answer_relevance(question, answer), 4)}}
