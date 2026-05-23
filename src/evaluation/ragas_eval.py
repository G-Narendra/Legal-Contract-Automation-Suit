"""
RAGAS-style evaluation for retrieval and generation quality.

Evaluates:
- Context precision: Are retrieved docs relevant?
- Context recall: Are all needed docs retrieved?
- Faithfulness: Does the answer stay grounded in retrieved docs?
- Answer relevance: How relevant is the answer to the query?
"""

from typing import Dict, List, Optional


def evaluate_retrieval(retrieved_docs: List[str],
                        relevant_docs: List[str]) -> Dict:
    """Evaluate retrieval quality.

    Args:
        retrieved_docs: Documents retrieved by the system
        relevant_docs: Ground truth relevant documents

    Returns:
        Dict with precision, recall, MRR scores
    """
    if not retrieved_docs or not relevant_docs:
        return {"precision": 0, "recall": 0, "mrr": 0}

    retrieved_set = set(d.lower().strip()[:100] for d in retrieved_docs)
    relevant_set = set(d.lower().strip()[:100] for d in relevant_docs)

    true_positives = len(retrieved_set & relevant_set)

    precision = true_positives / len(retrieved_set) if retrieved_set else 0
    recall = true_positives / len(relevant_set) if relevant_set else 0

    # Mean Reciprocal Rank
    mrr = 0
    for rank, doc in enumerate(retrieved_docs, 1):
        doc_key = doc.lower().strip()[:100]
        if any(doc_key in rel or rel in doc_key for rel in relevant_set):
            mrr = 1.0 / rank
            break

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(2 * precision * recall / (precision + recall), 3) if (precision + recall) > 0 else 0,
        "mrr": round(mrr, 3),
    }


def evaluate_faithfulness(claims: List[str],
                           context: List[str]) -> Dict:
    """Evaluate faithfulness: Are claims grounded in context?

    Each claim is checked against the context to see if it can be
    directly supported. Lower faithfulness = hallucination risk.
    """
    if not claims or not context:
        return {"faithfulness": 1.0, "supported_claims": 0, "total_claims": 0}

    context_text = " ".join(context).lower()
    supported = 0

    for claim in claims:
        claim_lower = claim.lower()
        # Simple substring check (in production, use NLI model)
        if any(phrase in context_text for phrase in claim_lower.split() if len(phrase) > 3):
            supported += 1

    return {
        "faithfulness": round(supported / len(claims), 3) if claims else 1.0,
        "supported_claims": supported,
        "total_claims": len(claims),
    }


def evaluate_answer_relevance(answer: str, query: str) -> Dict:
    """Evaluate answer relevance to the query.

    Checks if the answer directly addresses the question.
    """
    if not answer or not query:
        return {"relevance": 0}

    query_terms = set(query.lower().split())
    answer_lower = answer.lower()

    # What fraction of query terms appear in the answer?
    matched_terms = sum(1 for term in query_terms if term in answer_lower)

    relevance = matched_terms / len(query_terms) if query_terms else 0

    return {
        "relevance": round(relevance, 3),
        "query_terms": len(query_terms),
        "matched_terms": matched_terms,
    }
