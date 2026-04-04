import uuid
from qdrant.qdrant import QdrantHybridClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from models.model_loader import ModelLoader
from fastembed import SparseTextEmbedding
from utils.Data_ingestion import chunking

# Reuse the sparse model loader
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

def compute_sparse_vectors(texts):
    return list(sparse_model.embed(texts))

class ResumeEmbedder:
    def __init__(self, qdrant: QdrantHybridClient):
        self.model_loader = ModelLoader()
        self.embedding_model = self.model_loader.load_embedding_model()
        self.qdrant = qdrant
        self.collection = "resume_collection"
        self.chunker = chunking(chunk_size=500, stride=50)

    def _embed_chunks(self, texts):
        sparse = compute_sparse_vectors(texts)
        if hasattr(self.embedding_model, "embed_documents"):
            dense = self.embedding_model.embed_documents(texts)
        elif hasattr(self.embedding_model, "get_text_embedding_batch"):
            dense = self.embedding_model.get_text_embedding_batch(texts)
        else:
            dense = [self.embedding_model.get_text_embedding(t) for t in texts]

        return dense, sparse

    async def ingest(self, user_id: str, text: str):
        # Ensure collection exists before we try to put things in it!
        try:
            await self.qdrant.create_collection(self.collection)
        except Exception:
            pass

        # Check if this user's resume is already embedded in the database
        try:
            records, _ = await self.qdrant.client.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="user_id", match=MatchValue(value=user_id)
                        )
                    ]
                ),
                limit=1,
            )
            if records:
                # Resume is already inside DB, skip doing expensive embedding again!
                print(f"Resume for user {user_id} already exists in DB. Skipping ingestion.")
                return
        except Exception:
            pass

        # Chunk the resume text
        chunks_objs = self.chunker.chunk_text(text)
        texts = [chunk.text for chunk in chunks_objs]
        if not texts:
            return
            
        dense, sparse = self._embed_chunks(texts)

        points = []
        for d, s, t in zip(dense, sparse, texts):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "text_dense": d,
                        "bm25_sparse": {
                            "indices": s.indices.tolist() if hasattr(s.indices, "tolist") else s.indices,
                            "values": s.values.tolist() if hasattr(s.values, "tolist") else s.values
                        }
                    },
                    payload={
                        "user_id": user_id,
                        "text": t
                    }
                )
            )

        # Upsert the documents into Qdrant
        await self.qdrant.upsert(self.collection, points)

    async def search(self, user_id: str, topic: str, top_k=3):
        # Build query specific to User's resume search
        query_text = f"Resume context for {topic}"

        # 1. Embed query
        if hasattr(self.embedding_model, "embed_query"):
            query_dense = self.embedding_model.embed_query(query_text)
        elif hasattr(self.embedding_model, "get_query_embedding"):
            query_dense = self.embedding_model.get_query_embedding(query_text)
        else:
            query_dense = self.embedding_model.get_text_embedding(query_text)
            
        if hasattr(query_dense, "tolist"):  
            query_dense = query_dense.tolist()

        # 2. Get sparse embedding
        sparse_obj = compute_sparse_vectors([query_text])[0]
        query_sparse = {
            "indices": sparse_obj.indices.tolist() if hasattr(sparse_obj.indices, "tolist") else sparse_obj.indices,
            "values": sparse_obj.values.tolist() if hasattr(sparse_obj.values, "tolist") else sparse_obj.values,
        }

        # 3. Hybrid search in Qdrant
        results = await self.qdrant.search(
            collection_name=self.collection,
            query_dense=query_dense,
            query_sparse=query_sparse,
            top_k=top_k * 2,  # Fetch slightly more to filter out by user_id
        )

        filtered = []
        for point in results:
            payload = point.payload or {}
            # Filter specifically by User ID so they only get their own resume context
            if payload.get("user_id") != user_id:
                continue
                
            filtered.append(payload.get("text", ""))

        return filtered[:top_k]
