"""
xai.py — Explainable AI & Skill Gap Analysis Module

Responsibilities:
1. Skill coverage / overlap detection — extracts exact matched skills
   vs. missing JD skills.
2. Feature contribution score breakdown — decomposes the final ranking
   score into per-factor contributions so recruiters understand the
   candidate's exact rank justification.
3. SHAP-style visual score breakdown — generates waterfall-style
   decomposition data suitable for frontend rendering.

Adapted from anshvermadev/ResumeIQ.
"""

import logging
import math
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class XAIExplainer:
    """
    Explainability engine that supplies the frontend with:

    * Matched (green) and missing (red) skill badges.
    * Section-weighted feature contribution breakdown.
    * SHAP-style score decomposition for visualisation.
    """

    # Default section weights for the score breakdown
    SECTION_WEIGHTS = {
        "skills": 0.50,       # 50 % weight on skills
        "experience": 0.30,   # 30 % weight on experience
        "education": 0.20,    # 20 % weight on education
    }

    def __init__(self, weights: Optional[dict] = None):
        """
        Args:
            weights: Optional dict overriding section weights.
                     Default: ``{'skills': 0.50, 'experience': 0.30, 'education': 0.20}``
        """
        self.weights = weights or self.SECTION_WEIGHTS.copy()
        self._validate_weights()

    def _validate_weights(self):
        """Ensure weights sum to approximately 1.0."""
        expected = set(self.SECTION_WEIGHTS)
        if set(self.weights) != expected or any(
            not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight < 0
            for weight in self.weights.values()
        ):
            raise ValueError(
                "Section weights must define non-negative skills, experience, "
                "and education values."
            )
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("At least one section weight must be greater than zero.")
        if abs(total - 1.0) > 0.01:
            logger.warning("Weights sum to %.4f, normalising to 1.0", total)
            for key in self.weights:
                self.weights[key] /= total

    # ------------------------------------------------------------------ #
    #  Skill coverage analysis                                            #
    # ------------------------------------------------------------------ #
    def analyze_skill_coverage(self, resume_skills: list, jd_skills: list) -> dict:
        """
        Analyse skill coverage between resume and job description.

        Args:
            resume_skills: Skills extracted from the resume.
            jd_skills: Skills required in the job description.

        Returns:
            Dict with ``matched_skills``, ``missing_skills``,
            ``additional_skills``, ``coverage_percentage``,
            ``coverage_level``, ``total_jd_skills``, ``matched_count``,
            ``missing_count``.
        """
        resume_lower = set(s.lower().strip() for s in resume_skills if s)
        jd_lower = set(s.lower().strip() for s in jd_skills if s)
        unique_jd_skills = []
        seen_jd_skills = set()
        for skill in jd_skills:
            normalized = skill.lower().strip() if skill else ""
            if normalized and normalized not in seen_jd_skills:
                unique_jd_skills.append(skill)
                seen_jd_skills.add(normalized)

        matched_skills: List[str] = []
        missing_skills: List[str] = []

        for skill in unique_jd_skills:
            if skill and skill.lower().strip() in resume_lower:
                matched_skills.append(skill)
            elif skill:
                missing_skills.append(skill)

        # Skills in resume but not in JD
        additional_skills = []
        seen_resume_skills = set()
        for skill in resume_skills:
            normalized = skill.lower().strip() if skill else ""
            if normalized and normalized not in jd_lower and normalized not in seen_resume_skills:
                additional_skills.append(skill)
                seen_resume_skills.add(normalized)

        total_jd_skills = len(unique_jd_skills)
        matched_count = len(matched_skills)
        coverage_percentage = (
            matched_count / total_jd_skills * 100 if total_jd_skills > 0 else 0
        )

        if coverage_percentage >= 75:
            coverage_level = "high"
        elif coverage_percentage >= 40:
            coverage_level = "moderate"
        else:
            coverage_level = "low"

        return {
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "additional_skills": additional_skills,
            "coverage_percentage": round(coverage_percentage, 1),
            "coverage_level": coverage_level,
            "total_jd_skills": total_jd_skills,
            "matched_count": matched_count,
            "missing_count": len(missing_skills),
        }

    # ------------------------------------------------------------------ #
    #  Feature contribution breakdown                                      #
    # ------------------------------------------------------------------ #
    def generate_feature_breakdown(self, tfidf_score=None, sbert_score=None,
                                    skills_score=None, experience_score=None,
                                    education_score=None) -> dict:
        """
        Generate a feature contribution breakdown (SHAP-style) showing
        how each factor contributes to the final ranking score.

        Args:
            tfidf_score: TF-IDF similarity [0, 1].
            sbert_score: SBERT similarity [0, 1].
            skills_score: Skills match [0, 1].
            experience_score: Experience match [0, 1].
            education_score: Education match [0, 1].

        Returns:
            Dict with ``features`` (list of per-factor dicts),
            ``total_score``, ``summary``.
        """
        features: List[dict] = []
        total_score = 0.0

        # --- Weighted section contributions --- #

        if skills_score is not None:
            weight = self.weights["skills"]
            contribution = skills_score * weight
            features.append({
                "name": "Skills Match",
                "value": round(skills_score, 4),
                "weight": weight,
                "contribution": round(contribution, 4),
                "icon": "fa-check-circle",
                "color": (
                    "success" if skills_score >= 0.6
                    else ("warning" if skills_score >= 0.3 else "danger")
                ),
            })
            total_score += contribution

        if experience_score is not None:
            weight = self.weights["experience"]
            contribution = experience_score * weight
            features.append({
                "name": "Experience Match",
                "value": round(experience_score, 4),
                "weight": weight,
                "contribution": round(contribution, 4),
                "icon": "fa-briefcase",
                "color": (
                    "success" if experience_score >= 0.6
                    else ("warning" if experience_score >= 0.3 else "danger")
                ),
            })
            total_score += contribution

        if education_score is not None:
            weight = self.weights["education"]
            contribution = education_score * weight
            features.append({
                "name": "Education Match",
                "value": round(education_score, 4),
                "weight": weight,
                "contribution": round(contribution, 4),
                "icon": "fa-graduation-cap",
                "color": (
                    "success" if education_score >= 0.6
                    else ("warning" if education_score >= 0.3 else "danger")
                ),
            })
            total_score += contribution

        # --- Informational model scores (not part of weighted total) --- #

        if tfidf_score is not None:
            features.append({
                "name": "TF-IDF Similarity",
                "value": round(tfidf_score, 4),
                "weight": 0.0,
                "contribution": 0.0,
                "icon": "fa-search",
                "color": "info",
                "informational": True,
            })

        if sbert_score is not None:
            features.append({
                "name": "SBERT Similarity",
                "value": round(sbert_score, 4),
                "weight": 0.0,
                "contribution": 0.0,
                "icon": "fa-brain",
                "color": "info",
                "informational": True,
            })

        return {
            "features": features,
            "total_score": round(total_score, 4),
            "summary": self._generate_summary(features, total_score),
        }

    def _generate_summary(self, features: list, total_score: float) -> str:
        """Human-readable summary of the score breakdown."""
        weighted = [f for f in features if not f.get("informational", False)]
        if not weighted:
            return "No scoring data available."

        strongest = max(weighted, key=lambda x: x["contribution"])
        weakest = min(weighted, key=lambda x: x["contribution"])

        parts = [f"Overall match score: {round(total_score * 100, 1)}%."]
        parts.append(
            "Strongest factor: "
            f"{strongest['name']} "
            f"({round(strongest['contribution'] * 100, 1)}% contribution)."
        )
        if weakest["contribution"] < 0.1:
            parts.append(f"Area for improvement: {weakest['name']}.")

        return " ".join(parts)

    # ------------------------------------------------------------------ #
    #  SHAP-style waterfall decomposition                                  #
    # ------------------------------------------------------------------ #
    def generate_shap_style_breakdown(self, feature_scores: dict) -> Optional[dict]:
        """
        Generate a SHAP-style score decomposition suitable for frontend
        visualisation (waterfall chart data).

        Each feature's contribution is shown as a positive or negative
        deviation from the baseline (mean score).

        Args:
            feature_scores: ``{feature_name: score [0, 1]}``

        Returns:
            Dict with ``baseline``, ``contributions`` (list of per-factor
            dicts), ``final_score``.
        """
        if not feature_scores:
            return None

        scores = list(feature_scores.values())
        baseline = sum(scores) / len(scores) if scores else 0.0

        contributions: List[dict] = []
        cumulative = baseline

        for name, score in feature_scores.items():
            delta = score - baseline
            cumulative += delta
            contributions.append({
                "name": name,
                "score": round(score, 4),
                "baseline": round(baseline, 4),
                "delta": round(delta, 4),
                "cumulative": round(cumulative, 4),
                "positive": delta >= 0,
            })

        return {
            "baseline": round(baseline, 4),
            "contributions": contributions,
            "final_score": round(cumulative, 4),
        }

    # ------------------------------------------------------------------ #
    #  Full ranking explanation                                           #
    # ------------------------------------------------------------------ #
    def explain_ranking(self, resume_parsed_data: dict, jd_parsed_data: dict,
                        tfidf_score=None, sbert_score=None,
                        skills_score=None, experience_score=None,
                        education_score=None) -> dict:
        """
        Generate a complete explainability report for a single candidate.

        Combines skill coverage analysis with feature contribution
        breakdown to give recruiters a full picture of why a candidate
        ranked where they did.

        Returns:
            Dict with ``skill_coverage``, ``feature_breakdown``,
            ``shap_breakdown``, ``badges``, ``summary``.
        """
        # Skill coverage
        resume_skills = resume_parsed_data.get("skills", [])
        jd_skills = jd_parsed_data.get("skills", [])
        skill_coverage = self.analyze_skill_coverage(resume_skills, jd_skills)

        # Feature contribution breakdown
        feature_breakdown = self.generate_feature_breakdown(
            tfidf_score=tfidf_score,
            sbert_score=sbert_score,
            skills_score=skills_score,
            experience_score=experience_score,
            education_score=education_score,
        )

        # SHAP-style waterfall
        feature_scores: Dict[str, float] = {}
        if skills_score is not None:
            feature_scores["Skills"] = skills_score
        if experience_score is not None:
            feature_scores["Experience"] = experience_score
        if education_score is not None:
            feature_scores["Education"] = education_score
        shap_breakdown = self.generate_shap_style_breakdown(feature_scores)

        # UI badges
        badges = self._generate_badges(skill_coverage)

        return {
            "skill_coverage": skill_coverage,
            "feature_breakdown": feature_breakdown,
            "shap_breakdown": shap_breakdown,
            "badges": badges,
            "summary": feature_breakdown["summary"],
        }

    # ------------------------------------------------------------------ #
    #  Badge generation for templates                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _generate_badges(skill_coverage: dict) -> List[dict]:
        """Generate skill badge data for frontend rendering."""
        badges: List[dict] = []

        for skill in skill_coverage["matched_skills"]:
            badges.append({
                "skill": skill, "status": "matched",
                "class": "bg-success", "icon": "fa-check",
            })
        for skill in skill_coverage["missing_skills"]:
            badges.append({
                "skill": skill, "status": "missing",
                "class": "bg-danger", "icon": "fa-times",
            })
        for skill in skill_coverage["additional_skills"]:
            badges.append({
                "skill": skill, "status": "additional",
                "class": "bg-info", "icon": "fa-plus",
            })

        return badges
