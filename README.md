# ChunkReranker

## Overview

This project builds a small, reproducible question-answering service
over real machinery safety documents. It performs three key steps: (1)
baseline similarity search, (2) reranking for smarter evidence
selection, and (3) before/after comparison. The design is local, honest,
and deployable on any CPU.

----------------------------------------------------------------------------------------------------------------------------------

## Setup

1.  **Clone the repository and move into the project folder.**

2.  **Create a Python virtual environment:**

    ``` bash
    python -m venv venv
    source venv/bin/activate    # For Linux/Mac
    venv\Scripts\activate       # For Windows
    ```

3.  **Install dependencies:**

    ``` bash
    pip install -r requirements.txt
    ```

----------------------------------------------------------------------------------------------------------------------------------

## How to Run

1.  **Add all provided PDFs to the `data/` folder.**

2.  **Ingest and chunk data:**

    ``` bash
    python src/ingest.py
    ```

    Output: `data/chunks.jsonl`

3.  **Create embeddings and index:**

    ``` bash
    python src/embeddings.py
    ```

    Output: `data/chroma_db/`

4.  **Start the API:**

    ``` bash
    python src/api.py
    ```

5.  **Run the interactive CLI for before/after comparison:**

    ``` bash
    python src/query_client.py
    ```

    Enter questions one by one, see baseline (BEFORE) and reranked
    (AFTER) results.

----------------------------------------------------------------------------------------------------------------------------------

## Results Table Format (example)

    -- BEFORE (Similarity Search) --
    1. Score: 0.6753
       Source: EN_TechnicalguideNo10_REVF.pdf (Page 7)
       Text: Safety and functional safety The purpose of safety ...

    -- AFTER (Reranked) --
    1. Score: 0.8142
       Source: safebk-rm002_-en-p.pdf (Page 17)
       Text: MACHINERY SAFEBOOK 5 Safety related control systems ...

----------------------------------------------------------------------------------------------------------------------------------

## What I Learned

Working on this project demonstrated that dense embeddings alone
surface relevant evidence from domain texts, but a hybrid reranker
(combining semantic and keyword scoring) reliably boosts top-quality
answers.It can be emplyed to fine tune the retreival as per ones need, like keyword matching or stop word elimination or semantic quality enhancer. The approach stays fast and reproducible on local hardware, and
the before/after comparison is key to honestly diagnosing real QA
improvement versus baseline.

This becomes an important 

----------------------------------------------------------------------------------------------------------------------------------

## Deliverables

-   **All code**: ingest/chunk, embedding/index, baseline search,
    reranker, API.
-   **sources.json**: Provided source list for the project (assignment
    spec).
-   **questions.json**: My 8 evaluation questions along with the result of each question as query.(8 question.txt)
-   **Short README**: (this file).
-   **Two example curl requests** :Provide you with curl request for cmd and powershell terminals along with the result.(Curl.txt).

----------------------------------------------------------------------------------------------------------------------------------

## Author 

Ritik Awasthi
https://github.com/DrParticlePhysicist

----------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------