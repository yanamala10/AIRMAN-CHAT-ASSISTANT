import os
import re
import json
import time
import logging
from pypdf import PdfReader
from app.rag import retrieve, generate_answer

# Setup logging to track evaluation progress
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def extract_questions_from_pdf(pdf_path):
    """
    Extracts the 50 aviation questions from the Sample test questions PDF.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Could not find PDF at: {pdf_path}")
    
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    # Improved regex to handle the specific numbering in the document
    raw_parts = re.split(r'\n\s*\d+\.\s', text)
    
    # Filter fragments and keep only the question line
    questions = [q.strip().split('\n')[0] for q in raw_parts if len(q.strip()) > 10]
    
    return questions[:50]

def run_evaluation(questions):
    """
    Runs evaluation and calculates hit-rate, faithfulness, and hallucination rate.
    """
    results = []
    hits = 0
    hallucinations = 0
    REFUSAL_PHRASE = "This information is not available in the provided document(s)."

    print(f"Starting Evaluation for {len(questions)} questions...")

    for i, q in enumerate(questions):
        start_time = time.time()
        
        # 1. Retrieval
        retrieved_chunks = retrieve(q)
        has_retrieved = len(retrieved_chunks) > 0
        if has_retrieved:
            hits += 1
        
        # 2. Generation
        answer = generate_answer(q, retrieved_chunks)
        latency = time.time() - start_time
        
        # 3. Grounding Logic
        is_refusal = answer.strip() == REFUSAL_PHRASE
        is_hallucination = False
        # If no chunks were found but the system didn't refuse, it hallucinated
        if not has_retrieved and not is_refusal:
            is_hallucination = True
            hallucinations += 1

        results.append({
            "id": i + 1,
            "question": q,
            "retrieved": has_retrieved,
            "refused": is_refusal,
            "hallucination": is_hallucination,
            "latency": round(latency, 2),
            "answer": answer
        })
        
        logging.info(f"Q{i+1}: {'HIT' if has_retrieved else 'MISS'} | Hallucination: {is_hallucination}")

    total = len(questions)
    metrics = {
        "retrieval_hit_rate": round((hits / total) * 100, 2),
        "hallucination_rate": round((hallucinations / total) * 100, 2),
        "faithfulness_proxy": round(((total - hallucinations) / total) * 100, 2)
    }

    return results, metrics

if __name__ == "__main__":
    # Use raw string (r"") to avoid SyntaxWarning
    QUESTIONS_PDF = "AI_ML Technical Assessment/Sampletestquestions.pdf"
    
    try:
        test_questions = extract_questions_from_pdf(QUESTIONS_PDF)
        eval_data, final_metrics = run_evaluation(test_questions)

        output_file = "evaluation_report.json"
        with open(output_file, "w") as f:
            json.dump({"metrics": final_metrics, "results": eval_data}, f, indent=4)

        print("\n" + "="*40)
        print("FINAL EVALUATION METRICS")
        print("="*40)
        print(f"Retrieval Hit-Rate: {final_metrics['retrieval_hit_rate']}%")
        print(f"Hallucination Rate: {final_metrics['hallucination_rate']}%")
        print(f"Faithfulness Score: {final_metrics['faithfulness_proxy']}%")
        print("="*40)
        print(f"Detailed results saved to: {output_file}")

    except Exception as e:
        print(f"Evaluation Failed: {e}")