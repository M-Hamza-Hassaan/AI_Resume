from app.models.resume import Resume
from app.models.job_description import JobDescription
from app.nlp.vectorizer import ResumeVectorizer
from app.nlp.xai import XAIExplainer
from app.nlp.fairness import TextAnonymizer
from bson.objectid import ObjectId
from app import mongo

import logging
import re

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity as _sk_cosine
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    SentenceTransformer = None
    _sk_cosine = None
    logger.warning(
        "sentence-transformers not installed — SBERT matching disabled. "
        "Run: pip install sentence-transformers torch"
    )

SBERT_MODEL_NAME = "all-MiniLM-L6-v2"


# --------------------------------------------------------------------------- #
#  METHOD 1 — TF-IDF Baseline Matcher                                          #
# --------------------------------------------------------------------------- #
class TFIDFMatcher:
    def __init__(self):
        self.vectorizer = ResumeVectorizer(method="tfidf")

    def fit_corpus(self, texts: list):
        self.vectorizer.fit_corpus(texts, persist=False)

    def compute_similarity(self, resume_vector, jd_vector):
        return self.vectorizer.compute_similarity(resume_vector, jd_vector)

    def compute_text_similarity(self, resume_text: str, jd_text: str) -> float:
        if not self.vectorizer._is_fitted():
            self.fit_corpus([jd_text, resume_text])
        resume_vector = self.vectorizer.vectorizer.transform([resume_text])
        jd_vector = self.vectorizer.vectorizer.transform([jd_text])
        return self.vectorizer.compute_similarity(
            resume_vector.toarray()[0].tolist(),
            jd_vector.toarray()[0].tolist(),
        )


# --------------------------------------------------------------------------- #
#  METHOD 2 — Sentence-BERT Semantic Matcher                                   #
# --------------------------------------------------------------------------- #
class SBERTMatcher:

    def __init__(self, model_name: str = None):
        self.model_name = model_name or SBERT_MODEL_NAME
        self._model = None

    def _load_model(self):
        """Lazy-load the SBERT model on first use."""
        if self._model is None:
            if not SBERT_AVAILABLE:
                raise RuntimeError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers torch"
                )
            logger.info("Loading SBERT model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, text: str):
        model = self._load_model()
        truncated = " ".join(text.split()[:400])
        return model.encode([truncated], convert_to_numpy=True)[0]

    def compute_similarity(self, resume_text: str, jd_text: str) -> float:
        if not SBERT_AVAILABLE:
            raise RuntimeError(
                "SBERT scoring was requested but sentence-transformers is unavailable."
            )

        try:
            jd_embedding = self.encode(jd_text)
            resume_embedding = self.encode(resume_text)
            similarity = _sk_cosine(
                jd_embedding.reshape(1, -1),
                resume_embedding.reshape(1, -1),
            )[0][0]
            return float(max(0.0, min(1.0, similarity)))
        except Exception as e:
            logger.error("SBERT matching failed: %s", e)
            raise RuntimeError("SBERT matching failed.") from e

    def encode_batch(self, texts: list):
        model = self._load_model()
        truncated = [" ".join(t.split()[:400]) for t in texts]
        return model.encode(
            truncated, convert_to_numpy=True, show_progress_bar=False
        )


# --------------------------------------------------------------------------- #
#  UNIFIED RANKER                                                              #
# --------------------------------------------------------------------------- #
class ResumeRanker:
    """
    Unified resume ranker with dual TF-IDF + SBERT matching engine.

    Combines:
      * TF-IDF baseline matching (fast keyword matching)
      * Sentence-BERT semantic matching (deep semantic understanding)
      * Section-weighted scoring (Skills 50 %, Experience 30 %, Education 20 %)
      * XAI explainability (skill gap analysis + feature breakdown)
      * Fairness / anonymisation toggle (mask sensitive proxies before matching)

    Backward-compatible with the original ``ResumeRanker`` interface used
    by the Flask routes and templates.
    """

    def __init__(self, vectorizer_method: str = "tfidf",
                 use_sbert: bool = False,
                 toggle_anonymization: bool = False):
        """
        Args:
            vectorizer_method: Baseline vectoriser method.
            use_sbert: If True and SBERT is available, also compute SBERT
                       semantic similarity scores.
            toggle_anonymization: If True, mask candidate text before
                                  TF-IDF and SBERT matching.
        """
        self.vectorizer = ResumeVectorizer(method=vectorizer_method)
        self.tfidf_matcher = TFIDFMatcher()
        self.sbert_matcher = (
            SBERTMatcher() if (use_sbert and SBERT_AVAILABLE) else None
        )
        self.use_sbert = use_sbert and SBERT_AVAILABLE
        self.xai = XAIExplainer()
        self.anonymizer = TextAnonymizer(
            toggle_anonymization=toggle_anonymization
        )
        self.toggle_anonymization = toggle_anonymization

    # ------------------------------------------------------------------ #
    #  Main ranking method                                                 #
    # ------------------------------------------------------------------ #
    def rank_resumes_for_job(self, job_description_id: str) -> list:
        """
        Rank all resumes for a specific job description.

        Returns a ranked list with dual-model scores, XAI breakdowns,
        and skill coverage analysis for each candidate.
        """
        jd = JobDescription.get_by_id(job_description_id)
        if not jd:
            return []

        resumes = Resume.get_all()
        ranked_resumes: list = []
        jd_text = jd.parsed_data.get("full_text", jd.description or "")
        self._fit_tfidf_corpus(resumes, [jd])

        for resume in resumes:
            # Skip resumes without parsed data
            if not resume.parsed_data:
                continue

            # --- TF-IDF similarity in the shared request corpus --- #
            resume_text = resume.parsed_data.get("full_text", "")
            scoring_resume_text = (
                self.anonymizer.anonymize(resume_text)
                if self.toggle_anonymization else resume_text
            )
            tfidf_score = self.tfidf_matcher.compute_text_similarity(
                scoring_resume_text, jd_text
            )

            # --- SBERT similarity (computed on the fly) --- #
            sbert_score = None
            if self.use_sbert and self.sbert_matcher:
                sbert_score = self.sbert_matcher.compute_similarity(
                    scoring_resume_text, jd_text
                )

            # --- Skill matching --- #
            matching_skills, missing_skills = (
                self.vectorizer.get_matching_skills(
                    resume.parsed_data.get("skills", []),
                    jd.parsed_data.get("skills", []),
                )
            )

            # --- Section scores --- #
            skills_score = self._compute_skills_score(
                matching_skills, jd.parsed_data.get("skills", [])
            )
            experience_score = self._compute_experience_score(
                resume.parsed_data.get("experience", []),
                jd.parsed_data.get("experience", []),
                resume.parsed_data.get("years_of_experience"),
            )
            education_score = self._compute_education_score(
                resume.parsed_data.get("education", []),
                jd.parsed_data.get("education", []),
            )

            # --- Combined final score --- #
            combined_score = self._compute_combined_score(
                tfidf_score, sbert_score, skills_score,
                experience_score, education_score,
            )

            # --- User info --- #
            user_data = mongo.db.users.find_one(
                {"_id": ObjectId(resume.user_id)}
            )
            user_name = (
                user_data.get("name", "Unknown") if user_data else "Unknown"
            )

            # --- XAI explanation --- #
            xai_report = self.xai.explain_ranking(
                resume.parsed_data, jd.parsed_data,
                tfidf_score=tfidf_score,
                sbert_score=sbert_score,
                skills_score=skills_score,
                experience_score=experience_score,
                education_score=education_score,
            )

            ranked_resumes.append({
                "resume_id": resume.id,
                "user_id": resume.user_id,
                "user_name": user_name,
                "similarity_score": combined_score,
                "tfidf_score": tfidf_score,
                "sbert_score": sbert_score,
                "combined_score": combined_score,
                "matching_skills": matching_skills,
                "missing_skills": missing_skills,
                "skills_match_percentage": skills_score * 100,
                "skills_score": skills_score,
                "experience_score": experience_score,
                "education_score": education_score,
                "xai_report": xai_report,
            })

        # Sort by combined score (descending)
        ranked_resumes.sort(
            key=lambda x: x["similarity_score"], reverse=True
        )
        return ranked_resumes

    # ------------------------------------------------------------------ #
    #  Per-resume insights                                                #
    # ------------------------------------------------------------------ #
    def get_resume_insights(self, resume_id: str,
                            job_description_id: str) -> dict | None:
        """
        Get detailed insights for a specific resume and job description.

        Returns dual-model scores, section scores, XAI breakdown, and
        legacy education/experience match dicts for template compatibility.
        """
        resume = Resume.get_by_id(resume_id)
        jd = JobDescription.get_by_id(job_description_id)

        if not resume or not jd:
            return None

        self._fit_tfidf_corpus(Resume.get_all(), [jd])
        resume_text = resume.parsed_data.get("full_text", "")
        jd_text = jd.parsed_data.get("full_text", jd.description or "")
        scoring_resume_text = (
            self.anonymizer.anonymize(resume_text)
            if self.toggle_anonymization else resume_text
        )
        tfidf_score = self.tfidf_matcher.compute_text_similarity(
            scoring_resume_text, jd_text
        )

        # SBERT similarity
        sbert_score = None
        if self.use_sbert and self.sbert_matcher:
            sbert_score = self.sbert_matcher.compute_similarity(
                scoring_resume_text, jd_text
            )

        # Skill matching
        matching_skills, missing_skills = self.vectorizer.get_matching_skills(
            resume.parsed_data.get("skills", []),
            jd.parsed_data.get("skills", []),
        )

        # Section scores
        skills_score = self._compute_skills_score(
            matching_skills, jd.parsed_data.get("skills", [])
        )
        experience_score_val = self._compute_experience_score(
            resume.parsed_data.get("experience", []),
            jd.parsed_data.get("experience", []),
            resume.parsed_data.get("years_of_experience"),
        )
        education_score_val = self._compute_education_score(
            resume.parsed_data.get("education", []),
            jd.parsed_data.get("education", []),
        )

        # Legacy education / experience match (template compatibility)
        education_match = self._evaluate_education_match(
            resume.parsed_data.get("education", []),
            jd.parsed_data.get("education", []),
        )
        experience_match = self._evaluate_experience_match(
            resume.parsed_data.get("experience", []),
            jd.parsed_data.get("experience", []),
        )

        # Combined score
        combined_score = self._compute_combined_score(
            tfidf_score, sbert_score, skills_score,
            experience_score_val, education_score_val,
        )

        # Skills match percentage
        skills_match_percentage = skills_score * 100

        # XAI explanation
        xai_report = self.xai.explain_ranking(
            resume.parsed_data, jd.parsed_data,
            tfidf_score=tfidf_score,
            sbert_score=sbert_score,
            skills_score=skills_score,
            experience_score=experience_score_val,
            education_score=education_score_val,
        )

        return {
            "similarity_score": combined_score,
            "tfidf_score": tfidf_score,
            "sbert_score": sbert_score,
            "combined_score": combined_score,
            "matching_skills": matching_skills,
            "missing_skills": missing_skills,
            "skills_match_percentage": skills_match_percentage,
            "skills_score": skills_score,
            "experience_score": experience_score_val,
            "education_score": education_score_val,
            "education_match": education_match,
            "experience_match": experience_match,
            "xai_report": xai_report,
        }

    # ------------------------------------------------------------------ #
    #  Section score helpers                                               #
    # ------------------------------------------------------------------ #
    def _compute_skills_score(self, matching_skills: list,
                              jd_skills: list) -> float:
        """Compute a normalised skills match score [0, 1]."""
        unique_jd_skills = {
            skill.casefold().strip() for skill in jd_skills if skill.strip()
        }
        if not unique_jd_skills:
            return 0.0
        unique_matches = {
            skill.casefold().strip()
            for skill in matching_skills
            if skill.casefold().strip() in unique_jd_skills
        }
        return len(unique_matches) / len(unique_jd_skills)

    def _compute_experience_score(self, resume_experience: list,
                                  jd_experience: list,
                                  resume_years=None) -> float:
        """Compute a normalised experience match score [0, 1]."""
        if not jd_experience:
            return 0.5  # no requirement → neutral

        jd_exp_text = " ".join(exp.get("text", "") for exp in jd_experience)
        years_match = re.search(
            r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b",
            jd_exp_text,
            re.IGNORECASE,
        )

        if years_match:
            required_years = float(years_match.group(1))
            if required_years <= 0:
                return 1.0
            stated_years = resume_years
            if stated_years is None:
                durations = [
                    item.get("duration_years")
                    for item in resume_experience
                    if item.get("duration_years") is not None
                ]
                stated_years = max(durations) if durations else None
            if stated_years is None:
                return 0.5 if resume_experience else 0.0
            return min(1.0, max(0.0, float(stated_years) / required_years))

        # No explicit years requirement → credit for having experience
        return 1.0 if resume_experience else 0.0

    def _compute_education_score(self, resume_education: list,
                                 jd_education: list) -> float:
        """Compute a normalised education match score [0, 1]."""
        if not jd_education:
            return 0.5  # no requirement → neutral

        if not resume_education:
            return 0.0

        resume_edu_text = " ".join(
            edu.get("text", "") for edu in resume_education
        ).lower()
        jd_edu_text = " ".join(
            edu.get("text", "") for edu in jd_education
        ).lower()

        degree_keywords = [
            "bachelor", "master", "phd", "doctorate",
            "bsc", "msc", "ba", "ma",
        ]
        for kw in degree_keywords:
            if kw in jd_edu_text and kw in resume_edu_text:
                return 1.0

        return 0.5  # partial match

    # ------------------------------------------------------------------ #
    #  Combined score (section-weighted)                                  #
    # ------------------------------------------------------------------ #
    def _compute_combined_score(self, tfidf_score: float,
                                sbert_score: float | None,
                                skills_score: float,
                                experience_score: float,
                                education_score: float) -> float:
        """
        Compute a combined final score using section-weighted scoring.

        Text similarity (TF-IDF or blend of TF-IDF + SBERT) contributes 60 %.
        Section-weighted score (Skills 50 %, Exp 30 %, Edu 20 %) contributes 40 %.
        """
        weights = self.xai.weights

        # Blend text similarity if both models available
        if sbert_score is not None and self.use_sbert:
            text_sim = (tfidf_score + sbert_score) / 2.0
        else:
            text_sim = tfidf_score

        # Section-weighted score
        section_score = (
            skills_score * weights["skills"]
            + experience_score * weights["experience"]
            + education_score * weights["education"]
        )

        # Final blend: 60 % text similarity + 40 % section score
        combined = 0.6 * text_sim + 0.4 * section_score
        return float(max(0.0, min(1.0, combined)))

    # ------------------------------------------------------------------ #
    #  Legacy evaluators (template compatibility)                          #
    # ------------------------------------------------------------------ #
    def _evaluate_education_match(self, resume_education, jd_education):
        """Legacy education match evaluator for template compatibility."""
        if not jd_education:
            return {"match": True, "score": 1.0,
                    "details": "No specific education requirements"}

        if not resume_education:
            return {"match": False, "score": 0.0,
                    "details": "No education information provided"}

        resume_edu_text = " ".join(
            edu.get("text", "") for edu in resume_education
        ).lower()
        jd_edu_text = " ".join(
            edu.get("text", "") for edu in jd_education
        ).lower()

        degree_keywords = [
            "bachelor", "master", "phd", "doctorate",
            "bsc", "msc", "ba", "ma",
        ]
        for kw in degree_keywords:
            if kw in jd_edu_text and kw in resume_edu_text:
                return {"match": True, "score": 1.0,
                        "details": f"Found matching {kw} degree"}

        return {"match": False, "score": 0.5,
                "details": "Education requirements partially met"}

    def _evaluate_experience_match(self, resume_experience, jd_experience):
        """Legacy experience match evaluator for template compatibility."""
        if not jd_experience:
            return {"match": True, "score": 1.0,
                    "details": "No specific experience requirements"}

        if not resume_experience:
            return {"match": False, "score": 0.0,
                    "details": "No experience information provided"}

        jd_exp_text = " ".join(
            exp.get("text", "") for exp in jd_experience
        )
        years_match = re.search(r"(\d+)[\+]?\s+years?", jd_exp_text)

        if years_match:
            required_years = int(years_match.group(1))
            resume_years = len(resume_experience)
            if resume_years >= required_years:
                return {"match": True, "score": 1.0,
                        "details": f"Meets {required_years}+ years requirement"}
            score = resume_years / required_years
            return {"match": False, "score": score,
                    "details": f"Has {resume_years} years, needs {required_years}"}

        return {"match": True, "score": 0.7,
                "details": "Experience appears relevant"}

    # ------------------------------------------------------------------ #
    #  Text similarity (for fairness / counterfactual testing)            #
    # ------------------------------------------------------------------ #
    def compute_text_similarity(self, resume_text: str, jd_text: str) -> float:
        """
        Compute similarity from raw text strings.

        Uses TF-IDF text similarity as the primary method. This is used
        by the ``FairnessAuditor`` for counterfactual perturbation testing.
        """
        return self.tfidf_matcher.compute_text_similarity(resume_text, jd_text)

    def compute_score_for_text(
        self,
        resume_text: str,
        jd_text: str,
        skills_score: float,
        experience_score: float,
        education_score: float,
    ) -> float:
        """Score a text counterfactual using the same model blend as ranking."""
        tfidf_score = self.tfidf_matcher.compute_text_similarity(
            resume_text, jd_text
        )
        sbert_score = (
            self.sbert_matcher.compute_similarity(resume_text, jd_text)
            if self.sbert_matcher else None
        )
        return self._compute_combined_score(
            tfidf_score,
            sbert_score,
            skills_score,
            experience_score,
            education_score,
        )

    def _fit_tfidf_corpus(self, resumes, job_descriptions):
        """Fit one shared, request-local vocabulary for all ranking documents."""
        texts = []
        for item in resumes:
            parsed_data = item.parsed_data or {}
            text = parsed_data.get("full_text", "")
            if text:
                texts.append(text)
                if self.toggle_anonymization:
                    texts.append(self.anonymizer.anonymize(text))
        for item in job_descriptions:
            parsed_data = item.parsed_data or {}
            text = parsed_data.get("full_text", "")
            if text:
                texts.append(text)
        self.tfidf_matcher.fit_corpus(texts)
