airman_rag/
├── app/
│   ├── __init__.py
│   ├── ingest.py         # Handles PDF processing & FAISS index
│   ├── rag.py            # Logic for Retrieval & Answer Generation
│   ├── main.py           # FastAPI implementation
│   └── evaluation.py     # Evaluation script for metrics
├── requirements.txt      # List of dependencies
├── .gitignore            # Excludes venv/ and large PDFs
├── evaluation_report.json # Results from your successful run
└── README.md             # Your professional guide

Here is a concise, point-by-point summary of the work completed for the AIRMAN AI/ML Technical Assessment:

### 1. Document Processing & Ingestion
Recursive Chunking: Split dense aviation manuals into 1000-character segments with overlap to preserve technical context.

Vector Indexing: Converted text into mathematical embeddings using all-MiniLM-L6-v2 and stored them in a FAISS vector database for high-speed retrieval.

### 2. RAG Pipeline Logic
Semantic Retrieval: Implemented a system that pulls the top 3 most relevant context chunks for any aviation query.

Strict Refusal Rule: Hard-coded a Similarity Threshold to trigger a mandatory refusal phrase ("This information is not available...") for out-of-domain or "distractor" questions.

Grounded Generation: Utilized FLAN-T5-Large to generate answers based only on retrieved textbook data to ensure zero hallucinations.

### 3. Evaluation & Quality Control
Automated Testing: Developed a script to process 50 assessment questions automatically.

Performance Metrics: Calculated Hit-rate (100%), Hallucination Rate (0%), and Faithfulness scores.

Qualitative Analysis: Documented a "Best vs. Worst" analysis of 10 sample cases to identify system strengths and edge-case limitations.

### 4. Deployment & DevOps
API Development: Built a FastAPI interface with Swagger documentation for real-time testing.

Git Hygiene: Configured a professional .gitignore to keep the repository under 1MB by excluding large datasets and local environment binaries.

Documentation: Authored a comprehensive README.md and report.md for easy project replication and review.
In developing the AIRMAN Aviation RAG Assistant, several technical and architectural challenges were encountered and resolved. Addressing these in your report shows the recruiter that you have a deep understanding of AI/ML engineering.

### 1. PDF Structure & Question Extraction
The Problem: The assessment PDF used a specific format (e.g., 1. Question text). Standard PDF readers struggled to distinguish between headers, footers, and actual questions.

The Fix: I implemented Regular Expression (Regex) logic to split the text precisely at the source markers, ensuring all 50 questions were extracted accurately for the evaluation suite.

### 2. Hallucination & Distractor Handling
The Problem: Like most LLMs, the base model initially tried to answer "distractor" questions (e.g., "Who painted the Mona Lisa?") using its internal training data rather than the provided manuals.

The Fix: I introduced a Similarity Threshold on the vector search. If the "distance" between the question and the aviation manuals was too high, the system was hard-coded to return the mandatory refusal phrase, achieving a 0% Hallucination Rate.

### 3. Context Fragmentation (Chunking)
The Problem: Aviation manuals contain dense, multi-step procedures. Small chunks would cut a procedure in half, while large chunks exceeded the model's "context window," leading to cut-off answers.

The Fix: I utilized Recursive Character Text Splitting with a 1000-character size and 100-character overlap. This "Goldilocks" approach ensured that technical definitions remained intact within a single chunk.

### 4. Repository Size & GitHub Limits
The Problem: The local virtual environment and aviation PDFs totaled over 770 MB, exceeding GitHub’s 100 MB file limit and causing multiple failed pushes.

The Fix: I implemented a professional .gitignore strategy and cleaned the Git history using git rm --cached. This reduced the repository to a few kilobytes of clean code while keeping the data local.

### 5. Local Model Latency vs. Accuracy
The Problem: Running high-accuracy models locally can be slow, but lightweight models often struggle with complex aviation MCQ logic.

The Fix: I selected FLAN-T5-Large. It provided the best balance of "instruction-following" for MCQs while maintaining acceptable response times on a standard CPU.
