from search import DocumentRetriever
from rerank import hybrid_rerank

def print_results(label, results, score_key, top=5):
    print(f"\n-- {label} --\n")
    for i, r in enumerate(results[:top], 1):
        print(f"{i}. Score: {r[score_key]:.4f}")
        print(f"Source: {r['metadata']['pdf']} (Page {r['metadata']['page']})")
        snippet = r['text'][:300].replace('\n', ' ')
        print(f"Text: {snippet}...\n")

def main():
    retriever = DocumentRetriever()
    print("Welcome to Industrial Safety QA (type 'exit' to quit)\n")
    while True:
        question = input("Enter your question: ").strip()
        if not question or question.lower() in ('exit', 'quit'):
            break
        
        baseline_results = retriever.query(question, top_k=5)
        print_results("BEFORE (Similarity Search)", baseline_results, 'similarity')

        reranked_results = hybrid_rerank(question, baseline_results, alpha=0.7)
        print_results("AFTER (Reranked)", reranked_results, 'final_score')

        print("="*60 + "\n")

if __name__ == "__main__":
    main()
