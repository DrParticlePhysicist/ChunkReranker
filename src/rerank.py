import re
from collections import Counter

def simple_tokenize(text):
    tokens = re.findall(r"\w+", text.lower())
    return tokens

def bm25_score(query_tokens, doc_tokens, k1=1.5, b=0.75, avgdl=100):
    """
    Simple BM25 formula for scoring keyword relevance in doc.
    avgdl = average doc length (tune as needed)
    """
    doc_len = len(doc_tokens)
    freqs = Counter(doc_tokens)
    score = 0.0
    for q in set(query_tokens):
        f = freqs.get(q, 0)
        if f == 0:
            continue
        numer = f * (k1 + 1)
        denom = f + k1 * (1 - b + b * doc_len / avgdl)
        score += numer / denom
    return score

def hybrid_rerank(question, results, alpha=0.63):
    """
    inputs:
    - question: user query (str)
    - results: list of dicts with keys: 'text', 'similarity', 'metadata'
    - alpha: weight for vector similarity in final score

    returns:
    - same results list sorted by combined score
    """
    q_tokens = simple_tokenize(question)
    avgdl = sum(len(simple_tokenize(r['text'])) for r in results) / len(results)

    for r in results:
        d_tokens = simple_tokenize(r['text'])
        kw_score = bm25_score(q_tokens, d_tokens, avgdl=avgdl)
        r['final_score'] = alpha * r['similarity'] + (1 - alpha) * (kw_score / (kw_score + 1))

    results.sort(key=lambda x: x['final_score'], reverse=True)
    return results

# Example usage:
if __name__ == "__main__":
    sample_results = [
        {"text": "Machine safety refers to ...", "similarity": 0.85, "metadata": {"pdf": "doc1.pdf", "page": 12}},
        {"text": "Industrial hazards are ...", "similarity": 0.80, "metadata": {"pdf": "doc2.pdf", "page": 5}},
        # etc.
    ]
    question = "What is machine safety?"
    reranked = hybrid_rerank(question, sample_results)
    for i, r in enumerate(reranked):
        print(f"Rank {i+1}: Score={r['final_score']:.3f}\nText snippet: {r['text'][:200]}...\n")
