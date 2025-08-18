import os
import json

# Robust imports whether you run as a package (backendAPI.main:app) or from the folder
try:
    from backendAPI.utils import hybrid_retrieval
    from .generator import generator  # instance of QueryAwareCaseGenerator
except ImportError:
    from utils import hybrid_retrieval
    from generator import generator

# Load dataset with a safe, relative path
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "legal_cases.json")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    CASES = json.load(f)


def rag_response(case_text: str):

    # --- retrieval (old) ---
    retrieved = hybrid_retrieval(case_text, CASES, top_k=1)

    argument_texts = []
    for c in retrieved:
        text = (
            f"Using {c['case_type']} precedent "
            f"({c['citation']}, {c['year']}): {c['summary']}"
        )
        argument_texts.append(text)

    combined_argument = " ".join(argument_texts) if argument_texts else \
        "No directly relevant precedent found; relying on general principles."

    # --- generation (new) ---
    generated_sentence = generator.generate(case_text)

    # --- return combined result ---
    return {
        "argument": f"RAG Lawyer argues: {combined_argument}",
        "citations": [c.get("citation") for c in retrieved],
        "metadata": [c.get("metadata") for c in retrieved],
        "generated_case": generated_sentence,  # NEW field
    }
