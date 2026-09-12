import os
import numpy as np
from typing import List, Dict, Any
from groq import Groq

from rag.embedder import ProductionEmbedder
from rag.retriever import BM25Retriever
from rag.reranker import CrossEncoderReranker
from rag.evaluator import RAGTriadEvaluator

class HybridRAGPipeline:
    def __init__(self, documents: List[Dict[str, Any]], chunk_words: int = 100):
        self.llm_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.embedder = ProductionEmbedder()
        self.parent_docs = documents
        self.chunk_words = chunk_words
        self.child_chunks = []
        
        self._build_hierarchical_chunks()
        
        if self.child_chunks:
            self.vectors = np.array([self.embedder.embed(c["text"]) for c in self.child_chunks])
            self.sparse_index = BM25Retriever(self.child_chunks)
        
        self.reranker = CrossEncoderReranker()
        self.evaluator = RAGTriadEvaluator(self.embedder)

    def _build_hierarchical_chunks(self):
        """Dynamic word-based chunking configured from Streamlit interface."""
        chunk_id = 0
        for p_idx, doc in enumerate(self.parent_docs):
            words = doc["text"].split()
            for i in range(0, len(words), self.chunk_words):
                chunk_text = " ".join(words[i:i+self.chunk_words])
                if chunk_text.strip():
                    self.child_chunks.append({
                        "child_id": chunk_id,
                        "parent_id": p_idx,
                        "doc_id": doc["id"],
                        "text": chunk_text
                    })
                    chunk_id += 1

    def dense_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q_vec = self.embedder.embed(query)
        sims = np.dot(self.vectors, q_vec)
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [{"child_id": idx, "score": float(sims[idx]), "doc": self.child_chunks[idx]} for idx in top_indices]

    def reciprocal_rank_fusion(self, dense_res: List[Dict], sparse_res: List[Dict], k: int = 60) -> List[Dict]:
        rrf_scores = {}
        doc_map = {}

        def process_results(results):
            for rank, item in enumerate(results, start=1):
                cid = item["doc"]["child_id"]
                doc_map[cid] = item["doc"]
                if cid not in rrf_scores:
                    rrf_scores[cid] = 0.0
                rrf_scores[cid] += 1.0 / (k + rank)

        process_results(dense_res)
        process_results(sparse_res)

        merged = [{"doc": doc_map[cid], "rrf_score": score} for cid, score in rrf_scores.items()]
        return sorted(merged, key=lambda x: x["rrf_score"], reverse=True)

    def generate_llm_response(self, query: str, context: str, model_name: str, temperature: float = 0.7) -> str:
        system_prompt = (
            "You are a precise document assistant. Answer the user question based ONLY on the provided context. "
            "If the answer cannot be determined from the context, state clearly that you do not have enough information."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        # Model mapping for Groq compatibility
        model_map = {
            "GPT": "openai/gpt-oss-20b",
            "QWEN": "qwen/qwen3.8-27b",
            "GROQ": "groq/compound-mini"
        }
        print(f"selecting model name through map {model_name} gives {model_map.get(model_name)}")
        selected_groq_model = model_map.get(model_name, "GPT")

        chat_completion = self.llm_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=selected_groq_model,
            temperature=temperature,
        )
        return chat_completion.choices[0].message.content

    def query(self, query_str: str, model_name: str, temperature: float, top_k_parents: int = 3) -> Dict[str, Any]:
        if not self.child_chunks:
            return {
                "generated_answer": "No valid document text available to query.",
                "rag_triad_scores": {"context_relevance": 0, "faithfulness": 0, "answer_relevance": 0}
            }

        dense_candidates = self.dense_search(query_str, top_k=5)
        sparse_candidates = self.sparse_index.search(query_str, top_k=5)
        fused_candidates = self.reciprocal_rank_fusion(dense_candidates, sparse_candidates)
        reranked = self.reranker.rerank(query_str, fused_candidates[:10])

        # FIX: Collect child chunks instead of massive parent documents to prevent 400 Context Overflow
        context_chunks = []
        for item in reranked[:top_k_parents]:
            doc_id = item["doc"]["doc_id"]
            chunk_text = item["doc"]["text"]
            context_chunks.append(f"[Source: {doc_id}]\n{chunk_text}")

        # Combine only relevant top chunks for LLM context
        full_context = "\n\n---\n\n".join(context_chunks)

        # Truncate context if it exceeds safety limit (~12,000 chars)
        max_chars = 12000
        if len(full_context) > max_chars:
            full_context = full_context[:max_chars] + "\n...[Context truncated for token limits]"

        generated_answer = self.generate_llm_response(query_str, full_context, model_name=model_name, temperature=temperature)
        
        context_relevance = self.evaluator.evaluate_context_relevance(query_str, full_context)
        faithfulness = self.evaluator.evaluate_faithfulness(generated_answer, full_context)
        answer_relevance = self.evaluator.evaluate_answer_relevance(query_str, generated_answer)
        
        return {
            "query": query_str,
            "matched_child_chunks": [item["doc"]["text"] for item in reranked[:top_k_parents]],
            "resolved_parent_context": full_context,
            "generated_answer": generated_answer,
            "rag_triad_scores": {
                "context_relevance": round(context_relevance, 4),
                "faithfulness": round(faithfulness, 4),
                "answer_relevance": round(answer_relevance, 4)
            }
        }