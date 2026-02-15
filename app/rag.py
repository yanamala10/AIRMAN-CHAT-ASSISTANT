import faiss
import pickle
import numpy as np
import re
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from app.config import INDEX_PATH, META_PATH

# =====================================================
# Load Embedding Model
# =====================================================
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# =====================================================
# Load FAISS Index
# =====================================================
index = faiss.read_index(INDEX_PATH)

# =====================================================
# Load Documents + Metadata
# =====================================================
with open(META_PATH, "rb") as f:
    data = pickle.load(f)

documents = data["documents"]
metadata = data["metadata"]

# =====================================================
# Load LLM
# =====================================================
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = AutoModelForSeq2SeqLM.from_pretrained(model_name)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llm.to(DEVICE)
llm.eval()


# =====================================================
# RETRIEVAL
# =====================================================
def retrieve(query, top_k=8):

    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for idx in indices[0]:
        idx = int(idx)
        if 0 <= idx < len(documents):
            results.append({
                "text": documents[idx],
                "meta": metadata[idx]
            })

    return results


# =====================================================
# MCQ Detection
# =====================================================
def is_mcq(question):
    return bool(re.search(r"\bA\.\s|\bB\.\s|\bC\.\s|\bD\.\s", question))


def extract_options(question):
    pattern = r"(A\..*?)(?=B\.|C\.|D\.|$)|(B\..*?)(?=C\.|D\.|$)|(C\..*?)(?=D\.|$)|(D\..*)"
    matches = re.findall(pattern, question)
    options = []
    for match in matches:
        for group in match:
            if group:
                options.append(group.strip())
    return options


def clean_question(question):
    return re.split(r"\bA\.\s|\bB\.\s|\bC\.\s|\bD\.\s", question)[0].strip()


# =====================================================
# GENERATE ANSWER
# =====================================================
def generate_answer(question, retrieved_chunks):

    if not retrieved_chunks:
        return "No relevant information found in the documents."

    # Build context safely
    context = ""
    for chunk in retrieved_chunks:
        if len(context) + len(chunk["text"]) < 4000:
            context += chunk["text"] + "\n\n"

    # -------------------------------------------------
    # MCQ MODE
    # -------------------------------------------------
    if is_mcq(question):

        options = extract_options(question)
        clean_q = clean_question(question)

        prompt = f"""
You are an aviation technical assistant.

Answer the question using ONLY the context below.

Context:
{context}

Question:
{clean_q}

Answer:
"""

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(DEVICE)

        with torch.no_grad():
            outputs = llm.generate(
                **inputs,
                max_new_tokens=120,
                temperature=0.1,
                do_sample=False
            )

        generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Try matching answer to options
        for option in options:
            if generated_answer.lower() in option.lower():
                return option

        # If no exact match, return best guess
        return generated_answer

    # -------------------------------------------------
    # NORMAL QUESTION MODE
    # -------------------------------------------------
    else:

        prompt = f"""
You are an aviation technical assistant.

Answer the question strictly using the context below.
Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
"""

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(DEVICE)

        with torch.no_grad():
            outputs = llm.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=False
            )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        if len(answer) < 3:
            return "No clear answer found in the documents."

        return answer
