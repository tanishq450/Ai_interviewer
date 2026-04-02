from utils.Data_ingestion import Docloader
from utils.Data_ingestion import chunking
from qdrant.qdrant import QdrantHybridClient
from models.model_loader import ModelLoader
from fastembed import SparseTextEmbedding
from qdrant_client.models import PointStruct




# -------- Sparse Model --------
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")


def compute_sparse_vectors(texts):
    return list(sparse_model.embed(texts))


class ResumeEmbeddings:

    def __init__(self, qdrant: QdrantHybridClient):
        self.docloader = Docloader()
        self.chunking = chunking()
        self.model_loader = ModelLoader()
        self.qdrant = qdrant
        self.collection = "resume_collection"

    # -------- STEP 1: LOAD + CHUNK --------
    def _prepare_data(self, resume_path: str):
        text = self.docloader.load_pdf(resume_path)
        chunks = self.chunking.chunk_text(text)
        documents = self.chunking.convert_chunks(chunks)
        return documents

    # -------- STEP 2: EMBEDDING --------
    def _embed_documents(self, documents):

        texts = [doc.text for doc in documents]

        sparse = compute_sparse_vectors(texts)
        dense = self.model_loader.load_embedding_model().embed_documents(texts)

        return dense, sparse, texts

    # -------- STEP 3: UPSERT --------
    async def upsert_documents(self, dense, sparse, texts, user_id, domain):

        points = []

        for i, (t, d, s) in enumerate(zip(texts, dense, sparse)):

            points.append(
                PointStruct(
                    id=f"{user_id}_{i}",
                    vector={
                        "text_dense": d,
                        "bm25_sparse": {
                            "indices": s.indices.tolist(),
                            "values": s.values.tolist()
                        }
                    },
                    payload={
                        "text": t,
                        "user_id": user_id,
                        "domain": domain
                    }
                )
            )

    
        await self.qdrant.upsert(self.collection, points)

    # -------- STEP 4: SEARCH (RAG) --------
    async def search(self, user_id, query):

        # Dense embedding
        query_dense = self.model_loader.load_embedding_model().embed_query(query)

        if hasattr(query_dense, "tolist"):
            query_dense = query_dense.tolist()

        # Sparse embedding
        sparse_obj = compute_sparse_vectors([query])[0]
        query_sparse = {
            "indices": sparse_obj.indices.tolist(),
            "values": sparse_obj.values.tolist()
        }

        # Filter by user
        filter_query = {
            "must": [
                {"key": "user_id", "match": {"value": user_id}}
            ]
        }

        results = await self.qdrant.search(
            collection_name=self.collection,
            query_dense=query_dense,
            query_sparse=query_sparse,
            top_k=5
        )

        return [p.payload["text"] for p in results]



    
         