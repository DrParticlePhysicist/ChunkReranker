from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

CHROMA_PATH = "data/chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
TOP_K = 5

class DocumentRetriever:
    def __init__(self, chroma_path=CHROMA_PATH, embedding_model=EMBEDDING_MODEL_NAME):
        # Load embedding model
        self.embedder = SentenceTransformer(embedding_model)
        # Connect to persistent Chroma DB
        self.client = chromadb.PersistentClient(path=chroma_path, settings=Settings(is_persistent=True))
        self.collection = self.client.get_collection(name="industrial_docs")
    
    def query(self, question, top_k=TOP_K):
        # Embed the query question vector
        q_emb = self.embedder.encode([question], normalize_embeddings=True)
        # Query the collection for top-k closest vectors
        results = self.collection.query(
            query_embeddings=q_emb.tolist(),
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        hits = []
        for doc_text, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            hits.append({
                "text": doc_text,
                "metadata": meta,
                "similarity": 1 - dist  # cosine distance to similarity
            })
        return hits

if __name__ == "__main__":
    retriever = DocumentRetriever()
    question = input("Enter your question: ")
    results = retriever.query(question)
    print(f"\nTop {len(results)} results:\n")
    for i, hit in enumerate(results):
        print(f"Result #{i+1}: Similarity={hit['similarity']:.4f}")
        print(f"Source: {hit['metadata']['pdf']} Page: {hit['metadata']['page']}")
        print(f"Text snippet: {hit['text'][:600]}...\n")
