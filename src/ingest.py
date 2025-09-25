import os
import glob
import json
import re
from PyPDF2 import PdfReader

PDF_DIR = "data/"
OUTPUT_PATH = "data/chunks.jsonl"
MIN_CHUNK_WORDS = 50
MAX_CHUNK_WORDS = 1000

def recursive_chunk(paragraph, min_len=MIN_CHUNK_WORDS, max_len=MAX_CHUNK_WORDS):
    words = paragraph.split()
    if len(words) <= max_len:
        return [paragraph.strip()]
    else:
        sentences = re.split(r'(?<=[.!?]) +', paragraph)
        chunks = []
        current_chunk = []
        current_len = 0
        for sentence in sentences:
            sent_words = sentence.split()
            current_len += len(sent_words)
            current_chunk.append(sentence)
            if current_len >= min_len:
                chunks.append(" ".join(current_chunk).strip())
                current_chunk = []
                current_len = 0
        if current_chunk:
            chunks.append(" ".join(current_chunk).strip())
        return chunks

def pdf_to_chunks(pdf_path):
    reader = PdfReader(pdf_path)
    results = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            para = para.strip()
            if len(para.split()) < MIN_CHUNK_WORDS:
                continue
            sub_chunks = recursive_chunk(para)
            for chunk_index, sub_chunk in enumerate(sub_chunks):
                if len(sub_chunk.split()) > 10:
                    results.append({
                        "text": sub_chunk,
                        "pdf": os.path.basename(pdf_path),
                        "page": page_num + 1,
                        "chunk_index": chunk_index
                    })
    return results

def process_all_pdfs(pdf_folder, output_path):
    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    total_chunks = 0
    with open(output_path, "w", encoding="utf-8") as outfile:
        for pdf_file in pdf_files:
            chunks = pdf_to_chunks(pdf_file)
            total_chunks += len(chunks)
            for chunk in chunks:
                json.dump(chunk, outfile)
                outfile.write("\n")
    return total_chunks

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    total = process_all_pdfs(PDF_DIR, OUTPUT_PATH)
    print(f"Ingestion complete. Total chunks created: {total}")
    print(f"Output stored in: {OUTPUT_PATH}")
