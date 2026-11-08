"""
airesume NLP package.

Contains the resume parsing, dual-model ranking, explainability (XAI),
and fairness/bias mitigation modules.

Modules:
    parser     — spaCy NER + regex resume/JD text extraction
    vectorizer — TF-IDF vectorisation with MongoDB persistence
    ranker     — Dual TF-IDF + SBERT matching engine with section-weighted scoring
    xai        — Explainable AI: skill gap analysis + SHAP-style score breakdown
    fairness   — Text anonymisation + counterfactual bias auditing
"""
