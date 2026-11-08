"""
vectorizer.py — TF-IDF vectorisation with correct batch fitting.

TF-IDF must be fitted across a shared document corpus before transforming
documents. Ranking fits a request-local corpus; persisted vectors are retained
for compatibility with older data and non-ranking callers.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Absolute path relative to this file — works regardless of cwd
_MODELS_DIR = Path(__file__).parent / "models"
_TFIDF_PATH = _MODELS_DIR / "tfidf_vectorizer.pkl"


def _load_or_create_vectorizer() -> TfidfVectorizer:
    """Load a persisted TF-IDF vectorizer or create a fresh unfitted one."""
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if _TFIDF_PATH.exists():
        try:
            with open(_TFIDF_PATH, "rb") as f:
                vect = pickle.load(f)
            logger.debug("Loaded TF-IDF vectorizer from %s", _TFIDF_PATH)
            return vect
        except Exception as e:
            logger.warning("Could not load TF-IDF vectorizer: %s. Creating new.", e)

    return TfidfVectorizer(
        max_features=8000,     # larger vocab than before for better coverage
        stop_words="english",
        ngram_range=(1, 2),    # unigrams + bigrams
        sublinear_tf=True,     # apply log normalization to TF
        min_df=1,
    )


def _save_vectorizer(vect: TfidfVectorizer):
    """Persist the fitted TF-IDF vectorizer to disk."""
    try:
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(_TFIDF_PATH, "wb") as f:
            pickle.dump(vect, f)
        logger.debug("Saved TF-IDF vectorizer → %s", _TFIDF_PATH)
    except Exception as e:
        logger.warning("Could not save TF-IDF vectorizer: %s", e)


def _build_text(parsed_data: dict) -> str:
    """Combine all relevant parsed fields into a single text string."""
    parts = [
        parsed_data.get("full_text", ""),
        " ".join(parsed_data.get("skills", [])),
        " ".join(edu.get("text", "") for edu in parsed_data.get("education", [])),
        " ".join(exp.get("text", "") for exp in parsed_data.get("experience", [])),
    ]
    text = " ".join(p for p in parts if p).lower()
    return " ".join(text.split())  # normalise whitespace


class ResumeVectorizer:
    """
    TF-IDF vectoriser for resumes and job descriptions.

    Correct usage for cosine similarity:
      The TF-IDF vectorizer MUST be fitted on all documents (resumes + JDs)
      together so they share the same vocabulary and IDF weights.
      Fitting on single documents produces incompatible vectors.

    Explicit fitting persists the vectorizer by default. Ranking can disable
    persistence and fit a request-local shared vocabulary instead.
    """

    def __init__(self, method: str = "tfidf"):
        self.method = method
        self.vectorizer = _load_or_create_vectorizer()

    def _is_fitted(self) -> bool:
        """Return True if the vectorizer has a built vocabulary."""
        return hasattr(self.vectorizer, "vocabulary_")

    # ------------------------------------------------------------------ #
    #  Batch fit (call this before vectorizing individual documents)      #
    # ------------------------------------------------------------------ #
    def fit_corpus(self, texts: list, persist: bool = True):
        """
        Fit the TF-IDF vectorizer on a corpus of texts.

        Should be called with all resume + JD texts before computing
        individual vectors. Saves the fitted vectorizer unless persistence is
        disabled for a request-local corpus.

        Args:
            texts: List of pre-processed text strings.
        """
        texts = [text for text in texts if isinstance(text, str) and text.strip()]
        if not texts:
            return
        self.vectorizer.fit(texts)
        if persist:
            _save_vectorizer(self.vectorizer)
        logger.info("TF-IDF vectorizer fitted on %d documents", len(texts))

    # ------------------------------------------------------------------ #
    #  Single-document vectorisation                                       #
    # ------------------------------------------------------------------ #
    def vectorize_resume(self, resume_data: dict) -> list:
        """Vectorize resume parsed data. Returns a list for MongoDB storage."""
        text = _build_text(resume_data)
        return self._vectorize_text(text)

    def vectorize_job_description(self, jd_data: dict) -> list:
        """Vectorize JD parsed data. Returns a list for MongoDB storage."""
        text = _build_text(jd_data)
        return self._vectorize_text(text)

    def _vectorize_text(self, text: str) -> list:
        """Core vectorisation logic. Fits on single doc if not yet fitted."""
        if self.method == "tfidf":
            if not self._is_fitted():
                # Fallback: fit on this single document
                # (suboptimal — caller should call fit_corpus first)
                logger.warning(
                    "TF-IDF vectorizer not fitted on a corpus. "
                    "Fitting on a single document (suboptimal). "
                    "Call fit_corpus(all_texts) for accurate similarities."
                )
                self.vectorizer.fit([text])
                _save_vectorizer(self.vectorizer)

            vector = self.vectorizer.transform([text]).toarray()[0]
            return vector.tolist()

        return []

    # ------------------------------------------------------------------ #
    #  Similarity and skill matching                                       #
    # ------------------------------------------------------------------ #
    def compute_similarity(self, resume_vector: list, jd_vector: list) -> float:
        """Compute cosine similarity between two stored vectors."""
        if not resume_vector or not jd_vector:
            return 0.0

        rv = np.array(resume_vector).reshape(1, -1)
        jv = np.array(jd_vector).reshape(1, -1)

        # Vectors must have matching dimensions
        if rv.shape[1] != jv.shape[1]:
            logger.warning(
                "Vector dimension mismatch (%d vs %d) — returning 0.0. "
                "Re-upload resumes after re-fitting the vectorizer.",
                rv.shape[1], jv.shape[1]
            )
            return 0.0

        similarity = float(cosine_similarity(rv, jv)[0][0])
        return max(0.0, min(1.0, similarity))

    def get_matching_skills(self, resume_skills: list, jd_skills: list):
        """
        Compare resume skills against JD skills (case-insensitive).

        Returns (matching_skills, missing_skills).
        """
        resume_lower = {s.lower().strip() for s in resume_skills if s}
        matching, missing = [], []
        seen = set()

        for skill in jd_skills:
            normalized = skill.lower().strip() if skill else ""
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            if normalized in resume_lower:
                matching.append(skill)
            else:
                missing.append(skill)

        return matching, missing
