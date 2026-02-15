from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import retrieve, generate_answer
from app.ingest import ingest

app = FastAPI()


class QueryRequest(BaseModel):
    question: str
    debug: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def run_ingest():
    ingest()
    return {"message": "Ingestion complete"}


@app.post("/ask")
def ask(request: QueryRequest):

    retrieved = retrieve(request.question)
    answer = generate_answer(request.question, retrieved)

    citations = [
        {
            "document": c["meta"]["document"],
            "page": c["meta"]["page"]
        }
        for c in retrieved
    ]

    response = {
        "answer": answer,
        "citations": citations
    }

    if request.debug:
        response["retrieved_chunks"] = retrieved

    return response
