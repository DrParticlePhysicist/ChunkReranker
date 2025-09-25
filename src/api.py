from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import JSONResponse
from search import DocumentRetriever
from rerank import hybrid_rerank

app = FastAPI(title="Industrial Safety QA")

retriever = DocumentRetriever()

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    rerank_alpha: float = 0.7  # weighting between vector and BM25

@app.post("/query")
async def query_qa(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Empty question provided.")
    # Step 1: Retrieve baseline results
    results = retriever.query(request.question, top_k=request.top_k)
    # Step 2: Rerank results
    reranked = hybrid_rerank(request.question, results, alpha=request.rerank_alpha)
    # Step 3: Format response with text + metadata + score
    response = [
        {
            "text": r["text"],
            "source_pdf": r["metadata"]["pdf"],
            "page": r["metadata"]["page"],
            "score": r.get("final_score", r.get("similarity")),
        }
        for r in reranked
    ]
    return JSONResponse(content={"results": response})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
