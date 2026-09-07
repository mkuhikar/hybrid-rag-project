import math,os
from typing import List, Dict, Any
from groq import Groq
import numpy as np

# Mocking external heavy models for clean self-contained execution
from sentence_transformers import SentenceTransformer, CrossEncoder



class ProductionEmbedder:
    """Real dense vector embedder using Sentence Transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Loads a lightweight, standard 384-dimensional dense embedding model
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        # Generates normalized float vector array
        vector = self.model.encode(text, normalize_embeddings=True)
        return np.array(vector, dtype=np.float32)

class BM25Retriever:
    """Implements basic BM25 sparse keyword search."""
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

class CrossEncoderReranker:
    """Production Cross-Encoder reranking pipeline using Hugging Face CrossEncoder."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        # Loads a pre-trained Cross-Encoder model (e.g., BAAI/bge-reranker-base, ms-marco-MiniLM-L-6-v2)
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        # 1. Format candidate pairs for full cross-attention evaluation: (query, candidate_text)
        pairs = [(query, cand["doc"]["text"]) for cand in candidates]

        # 2. Compute true joint Cross-Attention scores across all query-document pairs
        scores = self.model.predict(pairs)

        # 3. Attach output scores back to candidate dictionary objects
        for cand, score in zip(candidates, scores):
            cand["cross_score"] = float(score)

        # 4. Sort descending by Cross-Encoder score and return top-k matches
        reranked = sorted(candidates, key=lambda x: x["cross_score"], reverse=True)
        return reranked[:top_k]

class RAGTriadEvaluator:
    """Evaluates pipeline quality across Context Relevance, Faithfulness, and Answer Relevance."""
    def __init__(self, embedder: ProductionEmbedder):
        self.embedder = embedder

    def evaluate_context_relevance(self, query: str, retrieved_context: str) -> float:
        """Measures lexical keyword coverage between the query and retrieved context."""
        q_words = set(query.lower().split())
        c_words = set(retrieved_context.lower().split())
        if not q_words:
            return 0.0
        overlap = q_words.intersection(c_words)
        return float(len(overlap) / len(q_words))

    def evaluate_faithfulness(self, response: str, retrieved_context: str) -> float:
        """Measures how faithfully the generated output grounds itself in retrieved context sentences."""
        response_sentences = [s.strip() for s in response.split(".") if s.strip()]
        if not response_sentences:
            return 0.0
            
        context_words = set(retrieved_context.lower().split())
        grounded_count = 0
        
        for sentence in response_sentences:
            s_words = set(sentence.lower().split())
            if len(s_words) == 0:
                continue
            # Fraction of words in the sentence present in context
            overlap_ratio = len(s_words.intersection(context_words)) / len(s_words)
            if overlap_ratio >= 0.5:  # Sentence is predominantly supported by context
                grounded_count += 1

        return float(grounded_count / len(response_sentences))

    def evaluate_answer_relevance(self, query: str, response: str) -> float:
        """Measures semantic similarity between user query vector and generated output vector."""
        q_vec = self.embedder.embed(query)
        r_vec = self.embedder.embed(response)
        # Cosine similarity on normalized vectors
        similarity = np.dot(q_vec, r_vec)
        return float(similarity)
class HybridRAGPipeline:
    def __init__(self, documents: List[Dict[str, Any]]):
        self.llm_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.embedder = ProductionEmbedder()
        self.parent_docs = documents
        self.child_chunks = []
        self._build_hierarchical_chunks()
        
        # Build Dense Index
        self.vectors = np.array([self.embedder.embed(c["text"]) for c in self.child_chunks])
        
        # Build Sparse Index
        self.sparse_index = BM25Retriever(self.child_chunks)
        self.reranker = CrossEncoderReranker()

        # Build Evaluator
        self.evaluator = RAGTriadEvaluator(self.embedder)

    def _build_hierarchical_chunks(self):
        """Splits parent documents into smaller child search chunks."""
        chunk_id = 0
        for p_idx, doc in enumerate(self.parent_docs):
            words = doc["text"].split()
            # Split into chunks of 15 words
            chunk_size = 15
            for i in range(0, len(words), chunk_size):
                chunk_text = " ".join(words[i:i+chunk_size])
                self.child_chunks.append({
                    "child_id": chunk_id,
                    "parent_id": p_idx,
                    "text": chunk_text
                })
                chunk_id += 1

    def dense_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q_vec = self.embedder.embed(query)
        # Cosine Similarity against normalized vectors
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
    def generate_llm_response(self, query: str, context: str) -> str:
        """Calls the Groq API to generate an answer grounded strictly in the retrieved context."""
        system_prompt = (
            "You are a precise technical assistant. Answer the user question based ONLY on the provided context. "
            "If the answer cannot be determined from the context, state clearly that you do not have enough information."
        )

        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        chat_completion = self.llm_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model="openai/gpt-oss-20b",
            temperature=0.0,  # Zero temperature for deterministic, factual grounding
        )

        return chat_completion.choices[0].message.content

    def query(self, query_str: str, top_k_parents: int = 2) -> Dict[str, Any]:
        # 1. Multi-Route Candidate Generation
        dense_candidates = self.dense_search(query_str, top_k=5)
        sparse_candidates = self.sparse_index.search(query_str, top_k=5)

        # 2. Reciprocal Rank Fusion
        fused_candidates = self.reciprocal_rank_fusion(
            dense_candidates, sparse_candidates
        )

        # 3. Cross-Encoder Reranking
        reranked = self.reranker.rerank(query_str, fused_candidates[:10])

        # 4. Resolve Unique Top Parent Documents
        retrieved_parents = []
        seen_parent_ids = set()

        for item in reranked:
            p_id = item["doc"]["parent_id"]
            if p_id not in seen_parent_ids:
                seen_parent_ids.add(p_id)
                retrieved_parents.append(self.parent_docs[p_id]["text"])

            if len(seen_parent_ids) >= top_k_parents:
                break

        # Combine resolved parent contexts for LLM generation
        full_context = "\n---\n".join(retrieved_parents)
        # 5. LLM Answer Generation
        generated_answer = self.generate_llm_response(query_str, full_context)
        
        # 6. Evaluate via RAG Triad Metrics
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

        # return {
        #     "matched_child_chunks": [item["doc"]["text"] for item in reranked[:top_k_parents]],
        #     "resolved_parent_context": full_context,
        #     "reranked_scores": [item["cross_score"] for item in reranked[:top_k_parents]],
        # }

# Execution Pipeline Test
if __name__ == "__main__":
    knowledge_base = [
    {
        "id": "DOC-101",
        "text": (
            "The operational temperature limit for Model-X server blocks is 85 degrees Celsius. "
            "Exceeding this limit triggers emergency power reduction and throttles clock speeds to prevent core damage. "
            "If sustained temperatures exceed 95 degrees Celsius for more than 30 seconds, the system initiates an automated hard shutdown. "
            "To reset thermal alarm states, administrator credentials and a physical safety check of the coolant loops are required."
        )
    },
    {
        "id": "DOC-102",
        "text": (
            "Product inventory part SKU-9921-X corresponds to the high-durability thermal paste used across Model-X cooling assemblies. "
            "This thermal interface material boasts a high thermal conductivity rating of 12.5 W/mK to maximize heat transfer efficiency. "
            "Engineers must replace this compound during every biennial overhaul or whenever CPU cold plates are unseated. "
            "Each 10g tube of SKU-9921-X is rated for up to 15 server socket reapplications."
        )
    },
    {
        "id": "DOC-103",
        "text": (
            "Model-X cooling systems utilize dual-stage liquid pump units designated under assembly code PUMP-404-HY. "
            "The secondary redundant pump automatically engages if coolant pressure drops below 1.2 bar. "
            "Routine maintenance requires bleeding air lock valves every 6 months to maintain optimal flow rates. "
            "Replacement impellers for PUMP-404-HY are stocked under inventory part number SKU-1104-M."
        )
    },
    {
        "id": "DOC-104",
        "text": (
            "Critical system diagnostics report hardware faults via standardized hex codes logged in the BMC firmware. "
            "Error Code E-4012 indicates a thermal throttle event triggered by sensor bank alpha on server block 1. "
            "Error Code E-4018 signals a coolant flow rate deficit or secondary pump failure across the primary heat exchanger. "
            "When encountering E-4018, technicians should immediately inspect intake filters for dust obstruction before replacing mechanical parts."
        )
    },
    {
        "id": "DOC-105",
        "text": (
            "The warranty and service lifecycle for Model-X chassis components spans a standard period of 36 months from deployment. "
            "Extended support packages include 24/7 on-site component replacement under SLA Class Gold. "
            "All structural rack rails, fan trays under SKU-3302-F, and power distribution units are fully covered during this timeframe. "
            "Failure to use certified thermal paste SKU-9921-X during maintenance voids the core processor hardware warranty."
        )
    }
]

    pipeline = HybridRAGPipeline(knowledge_base)
    result = pipeline.query(input("Enter your technical question: "), top_k_parents=2)
    
    print("--- HYBRID RAG RETRIEVAL SUCCESSFUL ---")
    print("Child Chunk Matched:", result["matched_child_chunks"])
    print("Resolved Parent Context:", result["resolved_parent_context"])
    print("Generated Answer:", result["generated_answer"])
    print("RAG Triad Scores:", result["rag_triad_scores"])