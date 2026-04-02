from utils.Data_ingestion import Docloader
from qdrant.qdrant import QdrantHybridClient
from models.model_loader import ModelLoader
from fastembed import SparseTextEmbedding
from utils.Data_ingestion import chunking


model = SparseTextEmbedding(model_name="Qdrant/bm25")

def compute_sparse_vectors(texts):
    return list(model.embed(texts))


from qdrant_client.models import PointStruct


class QuestionEmbeddings:

    def __init__(self, qdrant: QdrantHybridClient):
        self.chunking = chunking()
        self.model_loader = ModelLoader()
        self.qdrant = qdrant
        self.collection = "question_collection"

    # -------- PREPARE --------
    def _prepare_data(self, question_data):
        """
        Expect list of dict:
        [
          {"question": "...", "domain": "...", "topic": "...", "difficulty": "..."}
        ]
        """
        return question_data

    # -------- EMBED --------
    def _embed_documents(self, questions):

        texts = [q["question"] for q in questions]

        sparse = compute_sparse_vectors(texts)
        dense = self.model_loader.load_embedding_model().embed_documents(texts)

        return dense, sparse, questions

    # -------- UPSERT --------
    async def upsert_documents(self, dense, sparse, questions):

        points = []

        for i, (q, d, s) in enumerate(zip(questions, dense, sparse)):

            points.append(
                PointStruct(
                    id=i,
                    vector={
                        "text_dense": d,
                        "bm25_sparse": {
                            "indices": s.indices.tolist(),
                            "values": s.values.tolist()
                        }
                    },
                    payload={
                        "question": q["question"],
                        "domain": q["domain"],
                        "topic": q["topic"],
                        "difficulty": q["difficulty"]
                    }
                )
            )

        # ✅ single call
        await self.qdrant.upsert(self.collection, points)

    # -------- SEARCH --------
    async def search(self, domain, topic, difficulty, query):

        # dense
        query_dense = self.model_loader.load_embedding_model().embed_query(query)

        if hasattr(query_dense, "tolist"):
            query_dense = query_dense.tolist()

        # sparse
        sparse_obj = compute_sparse_vectors([query])[0]
        query_sparse = {
            "indices": sparse_obj.indices.tolist(),
            "values": sparse_obj.values.tolist()
        }

        filter_query = {
            "must": [
                {"key": "domain", "match": {"value": domain}},
                {"key": "topic", "match": {"value": topic}},
                {"key": "difficulty", "match": {"value": difficulty}}
            ]
        }

        results = await self.qdrant.search(
            collection_name=self.collection,
            query_dense=query_dense,
            query_sparse=query_sparse,
            top_k=5
        )

        return [p.payload["question"] for p in results]