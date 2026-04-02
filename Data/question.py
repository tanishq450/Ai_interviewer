import uuid
from qdrant.qdrant import QdrantHybridClient
from qdrant_client.models import PointStruct
from models.model_loader import ModelLoader
from fastembed import SparseTextEmbedding


# -------- Sparse Model --------
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")


def compute_sparse_vectors(texts):
    return list(sparse_model.embed(texts))  


class QuestionEmbeddings:

    def __init__(self, qdrant: QdrantHybridClient):
        self.model_loader = ModelLoader()
        self.embedding_model = self.model_loader.load_embedding_model()
        self.qdrant = qdrant
        self.collection = "question_collection"

    def _embed_documents(self, questions):

        texts = [q["question"] for q in questions]

        sparse = compute_sparse_vectors(texts)
        if hasattr(self.embedding_model, "embed_documents"):
            dense = self.embedding_model.embed_documents(texts)
        elif hasattr(self.embedding_model, "get_text_embedding_batch"):
            dense = self.embedding_model.get_text_embedding_batch(texts)
        else:
            dense = [self.embedding_model.get_text_embedding(t) for t in texts]

        return dense, sparse, questions

    async def upsert_documents(self, dense, sparse, questions):

        points = []

        for q, d, s in zip(questions, dense, sparse):

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),  # FIXED
                    vector={
                        "text_dense": d,
                        "bm25_sparse": {
                            "indices": s.indices.tolist() if hasattr(s.indices, "tolist") else s.indices,
                            "values": s.values.tolist() if hasattr(s.values, "tolist") else s.values
                        }
                    },
                    payload={
                        "question": q.get("question", ""),
                        "domain": q.get("domain", ""),
                        "topic": q.get("topic", ""),
                        "difficulty": q.get("difficulty", "")
                    }
                )
            )

        await self.qdrant.upsert(self.collection, points)

    async def ingest(self, question_data):
        dense, sparse, questions = self._embed_documents(question_data)
        await self.upsert_documents(dense, sparse, questions)

    async def search(self, domain=None, topic=None, difficulty=None, query="", top_k=5):
        query_text = query or "interview question"

        if hasattr(self.embedding_model, "embed_query"):
            query_dense = self.embedding_model.embed_query(query_text)
        elif hasattr(self.embedding_model, "get_query_embedding"):
            query_dense = self.embedding_model.get_query_embedding(query_text)
        else:
            query_dense = self.embedding_model.get_text_embedding(query_text)
        if hasattr(query_dense, "tolist"):  
            query_dense = query_dense.tolist()

        sparse_obj = compute_sparse_vectors([query_text])[0]
        query_sparse = {
            "indices": sparse_obj.indices.tolist() if hasattr(sparse_obj.indices, "tolist") else sparse_obj.indices,
            "values": sparse_obj.values.tolist() if hasattr(sparse_obj.values, "tolist") else sparse_obj.values,
        }

        results = await self.qdrant.search(
            collection_name=self.collection,
            query_dense=query_dense,
            query_sparse=query_sparse,
            top_k=top_k,
        )

        filtered = []
        for point in results:
            payload = point.payload or {}
            if domain and payload.get("domain") not in (domain, "", None):
                continue
            if topic and payload.get("topic") not in (topic, "general", "", None):
                continue
            if difficulty and payload.get("difficulty") not in (difficulty, "", None):
                continue
            q = payload.get("question")
            if q:
                filtered.append(q)

        return filtered[:top_k]