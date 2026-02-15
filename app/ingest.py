import os
import re
import faiss
import pickle
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from app.config import DATA_PATH, INDEX_PATH, META_PATH

# =====================================================
# Config
# =====================================================
CHUNK_SIZE = 700
CHUNK_OVERLAP = 150

# =====================================================
# Load Embedding Model
# =====================================================
model = SentenceTransformer("all-MiniLM-L6-v2")


# =====================================================
# Clean Text
# =====================================================
def clean_text(text):

    text = re.sub(r"\s+", " ", text)   # remove extra spaces
    text = re.sub(r"\n+", "\n", text)  # normalize newlines
    text = text.strip()

    return text


# =====================================================
# Sliding Window Chunking (STRONGER)
# =====================================================
def chunk_text(text):

    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# =====================================================
# Get All PDFs
# =====================================================
def get_all_pdfs(folder_path):

    pdf_files = []

    if not os.path.exists(folder_path):
        print(f"❌ DATA_PATH does not exist: {folder_path}")
        return pdf_files

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                if "assignment" in file.lower() or "sample" in file.lower():
                    continue
                pdf_files.append(os.path.join(root, file))

    return pdf_files


# =====================================================
# INGEST
# =====================================================
def ingest():

    print(f"📂 DATA_PATH: {DATA_PATH}")

    documents = []
    metadata = []

    pdf_files = get_all_pdfs(DATA_PATH)
    print(f"📄 Found {len(pdf_files)} PDFs")

    if len(pdf_files) == 0:
        print("⚠️ No PDFs found.")
        return

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):

        file_name = os.path.basename(pdf_path)

        try:
            reader = PdfReader(pdf_path)
        except Exception as e:
            print(f"❌ Error reading {file_name}: {e}")
            continue

        for page_num, page in enumerate(reader.pages):

            try:
                raw_text = page.extract_text()
            except:
                continue

            if not raw_text:
                continue

            text = clean_text(raw_text)
            chunks = chunk_text(text)

            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                metadata.append({
                    "document": file_name,
                    "page": page_num + 1,
                    "chunk_id": f"{file_name}_p{page_num+1}_c{i}"
                })

    if not documents:
        print("⚠️ No text extracted.")
        return

    print(f"🔢 Total chunks created: {len(documents)}")

    # =====================================================
    # Create Embeddings
    # =====================================================
    embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True
    )

    embeddings = np.array(embeddings).astype("float32")

    # =====================================================
    # Create FAISS Index
    # =====================================================
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # =====================================================
    # Save Vectorstore
    # =====================================================
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump({
            "documents": documents,
            "metadata": metadata
        }, f)

    print("✅ Ingestion complete!")
    print(f"📊 Total vectors stored: {index.ntotal}")


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    ingest()
