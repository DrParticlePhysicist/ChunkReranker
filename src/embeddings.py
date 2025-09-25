import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
import os

CHUNKS_PATH = "data/chunks.jsonl"   # From ingest.py output
CHROMA_PATH = "data/chroma_db"      # Persistent vector DB path

def load_chunks(chunks_path):
    with open(chunks_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def embed_chunks(chunks, model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"):
    print("Loading embedding model...")
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32, normalize_embeddings=True)
    print("Embedding complete.")
    return embeddings

def store_in_chroma(chunks, embeddings, chroma_path):
    print("Initializing ChromaDB...")
    os.makedirs(chroma_path, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_path, settings=Settings(is_persistent=True))
    collection = client.get_or_create_collection(name="industrial_docs")
    print(f"Inserting {len(chunks)} chunks into ChromaDB...")
    for i, (chunk, embedding) in enumerate(tqdm(zip(chunks, embeddings), total=len(chunks))):
        metadata = {k: v for k, v in chunk.items() if k != "text"}
        collection.add(
            documents=[chunk["text"]],
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
            ids=[f"{metadata['pdf']}_pg{metadata['page']}_chunk{i}"]
        )
    print("Insertion complete.")

if __name__ == "__main__":
    print("Loading chunks...")
    chunks = load_chunks(CHUNKS_PATH)
    print(f"Total chunks loaded: {len(chunks)}")
    embeddings = embed_chunks(chunks)
    store_in_chroma(chunks, embeddings, CHROMA_PATH)
    print(f"All embeddings stored in {CHROMA_PATH}")
