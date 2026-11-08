import re
import logging
import math
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import spacy

    _nlp_fairness = None  #lazy-loaded singleton

    def _get_nlp():
        """Return a cached spaCy NLP instance for NER-based masking."""
        global _nlp_fairness
        if _nlp_fairness is None:
            try:
                _nlp_fairness = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning(
                    "en_core_web_sm is not installed; NER-based fairness masking "
                    "is unavailable. Install it with: python -m spacy download "
                    "en_core_web_sm"
                )
                _nlp_fairness = False
        return _nlp_fairness
except ImportError:  # pragma: no cover
    logger.warning("spaCy not available — fairness NER masking will be limited")
    _nlp_fairness = None
    _get_nlp = None


class TextAnonymizer:


    PRONOUNS = [
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "mr", "mr.", "mrs", "mrs.", "ms", "ms.",
        "miss", "sir", "madam",
    ]

    # Words that signal a college / university name
    COLLEGE_INDICATORS = [
        "university", "college", "institute", "education", "school",
        "academy", "polytechnic",
    ]

    def __init__(self, toggle_anonymization: bool = True):
        self.toggle_anonymization = toggle_anonymization
        self.mask_log: List[tuple] = []

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #
    def anonymize(self, text: str) -> str:
        
        if not self.toggle_anonymization or not text:
            return text

        self.mask_log = []
        anonymized = text

        anonymized = self._mask_person_names(anonymized)
        anonymized = self._mask_dates(anonymized)
        anonymized = self._mask_pronouns(anonymized)
        anonymized = self._mask_college_names(anonymized)
        anonymized = self._mask_locations(anonymized)

        return anonymized

    def get_mask_summary(self) -> Dict[str, list]:
        """Return a summary of every token that was masked, keyed by type."""
        summary: Dict[str, list] = {}
        for proxy_type, original, _replacement in self.mask_log:
            summary.setdefault(proxy_type, []).append(original) #Look up a proxy_type in the summary dictionary; if it doesn't exist yet, create it with an empty list [], and then add original to that list
        return summary

    
    def _mask_person_names(self, text: str) -> str:
        """Mask person names detected by spaCy NER."""
        if _get_nlp is None or _get_nlp() is False:
            return text
        doc = _get_nlp()(text)

        # The Traditional Way (Equivalent Code)
        person_spans = []

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Create a tuple of the start, end, and text, then add it to the list
                person_spans.append((ent.start_char, ent.end_char, ent.text))
        for start, end, original in sorted(person_spans, reverse=True):
            text = text[:start] + "[CANDIDATE_NAME]" + text[end:]
            self.mask_log.append(("NAME", original, "[CANDIDATE_NAME]"))

        return text

    def _mask_dates(self, text: str) -> str:
        def _year_repl(m):
            self.mask_log.append(("DATE", m.group(0), "[GRADUATION_YEAR]"))
            return "[GRADUATION_YEAR]"

        text = re.sub(r"\b(19|20)\d{2}\b", _year_repl, text)

        month_alt = (
            "January|February|March|April|May|June|July|"
            "August|September|October|November|December"
        )

        def _month_repl(match):
            self.mask_log.append(("DATE", match.group(0), "[DATE]"))
            return "[DATE]"

        text = re.sub(rf"\b(?:{month_alt})\b", _month_repl, text, flags=re.IGNORECASE)
        return text

    def _mask_pronouns(self, text: str) -> str:
        """Mask gender pronouns and titles (he, she, Mr, Mrs …)."""
        for pronoun in self.PRONOUNS:
            pattern = r"\b" + re.escape(pronoun) + r"\b"

            def _pronoun_repl(m):
                self.mask_log.append(("PRONOUN", m.group(0), "[PRONOUN]"))
                return "[PRONOUN]"

            text = re.sub(pattern, _pronoun_repl, text, flags=re.IGNORECASE)
        return text

    def _mask_college_names(self, text: str) -> str:
        """Mask college / university names (NER-based, with regex fallback)."""
        if _get_nlp is None or _get_nlp() is False:
            # Fallback: regex
            pattern = (
                r"\b[A-Z][\w\s]+(?:"
                + "|".join(self.COLLEGE_INDICATORS)
                + r")\b"
            )

            def _college_repl(m):
                self.mask_log.append(
                    ("COLLEGE_NAME", m.group(0), "[COLLEGE_NAME]")
                )
                return "[COLLEGE_NAME]"

            return re.sub(pattern, _college_repl, text, flags=re.IGNORECASE)

        doc = _get_nlp()(text)

        org_spans = []
        for ent in doc.ents:
            if ent.label_ == "ORG":
                ent_lower = ent.text.lower()
                if any(ind in ent_lower for ind in self.COLLEGE_INDICATORS):
                    org_spans.append((ent.start_char, ent.end_char, ent.text))

        for start, end, original in sorted(org_spans, reverse=True):
            text = text[:start] + "[COLLEGE_NAME]" + text[end:]
            self.mask_log.append(("COLLEGE_NAME", original, "[COLLEGE_NAME]"))

        return text

    def _mask_locations(self, text: str) -> str:
        """Mask geographic locations (cities, states, countries)."""
        if _get_nlp is None or _get_nlp() is False:
            return text

        doc = _get_nlp()(text)

        gpe_spans = [
            (ent.start_char, ent.end_char, ent.text)
            for ent in doc.ents
            if ent.label_ == "GPE"
        ]
        for start, end, original in sorted(gpe_spans, reverse=True):
            text = text[:start] + "[LOCATION]" + text[end:]
            self.mask_log.append(("LOCATION", original, "[LOCATION]"))

        return text


class FairnessAuditor:
    """
    Runs fairness audits on the matching pipeline.

    Two key tests:

    1. **Counterfactual Perturbation** — creates an anonymised version of a
       resume (all sensitive proxies masked), then compares the matching
       score against the original::

           Delta = Score(CV_original) - Score(CV_anonymised)

       A non-zero Delta signals model reliance on protected proxies.

    2. **Disparate Impact (4/5th Rule)** — checks whether the selection rate
       for any protected group falls below 80 % of the rate for the favoured
       group.
    """

    def __init__(self, ranker=None):
        """
        Args:
            ranker: A ``ResumeRanker`` (or any object with a
                    ``compute_text_similarity`` method) used for scoring.
        """
        self.ranker = ranker
        self.anonymizer = TextAnonymizer(toggle_anonymization=True)

    # ------------------------------------------------------------------ #
    #  Counterfactual test                                                 #
    # ------------------------------------------------------------------ #
    def counterfactual_test(self, resume_text: str, jd_text: str,
                            matching_fn=None) -> Optional[dict]:
        """
        Run a counterfactual perturbation test on a single resume.

        Args:
            resume_text: Original resume text.
            jd_text: Job description text.
            matching_fn: Optional ``callable(resume, jd) → float``.
                         If *None* the ``self.ranker`` is used.

        Returns:
            Dict with keys: ``original_score``, ``anonymised_score``,
            ``delta``, ``bias_detected``, ``bias_threshold``,
            ``mask_summary``, ``interpretation``.
        """
        anonymised_text = self.anonymizer.anonymize(resume_text)
        mask_summary = self.anonymizer.get_mask_summary()

        # Score both versions
        if matching_fn is not None:
            original_score = matching_fn(resume_text, jd_text)
            anonymised_score = matching_fn(anonymised_text, jd_text)
        elif self.ranker is not None:
            original_score = self.ranker.compute_text_similarity(
                resume_text, jd_text
            )
            anonymised_score = self.ranker.compute_text_similarity(
                anonymised_text, jd_text
            )
        else:
            logger.warning(
                "No matching function or ranker provided — cannot run test"
            )
            return None

        delta = original_score - anonymised_score
        bias_threshold = 0.05
        bias_detected = abs(delta) > bias_threshold

        return {
            "original_score": float(original_score),
            "anonymised_score": float(anonymised_score),
            "delta": float(delta),
            "bias_detected": bias_detected,
            "bias_threshold": bias_threshold,
            "mask_summary": mask_summary,
            "interpretation": self._interpret_delta(delta, bias_threshold),
        }

    # ------------------------------------------------------------------ #
    #  Disparate impact (4/5th rule)                                      #
    # ------------------------------------------------------------------ #
    def disparate_impact_test(self, scores_by_group: dict,
                              selection_threshold: float = 0.5) -> Optional[dict]:
        """
        Run the 4/5th Rule (Disparate Impact) test.

        Args:
            scores_by_group: ``{group_name: [score, score, …]}``.
            selection_threshold: Score above which a candidate is
                                 considered "selected".

        Returns:
            Dict with: ``selection_rates``, ``favored_group``,
            ``di_ratios``, ``four_fifths_threshold``, ``violation``,
            ``interpretation``.
        """
        if (
            not isinstance(selection_threshold, (int, float))
            or not math.isfinite(selection_threshold)
            or not 0 <= selection_threshold <= 1
        ):
            raise ValueError("selection_threshold must be a finite value in [0, 1].")

        selection_rates: Dict[str, float] = {}
        for group, scores in scores_by_group.items():
            if not scores:
                continue
            if any(
                not isinstance(score, (int, float))
                or not math.isfinite(score)
                or not 0 <= score <= 1
                for score in scores
            ):
                raise ValueError(
                    f"Scores for group {group!r} must be finite values in [0, 1]."
                )
            selected = sum(1 for s in scores if s >= selection_threshold)
            selection_rates[group] = selected / len(scores)

        if not selection_rates:
            return None

        favored_group = max(selection_rates, key=selection_rates.get)
        favored_rate = selection_rates[favored_group]

        di_ratios: Dict[str, float] = {}
        for group, rate in selection_rates.items():
            di_ratios[group] = rate / favored_rate if favored_rate > 0 else 0.0

        violation = favored_rate > 0 and any(
            r < 0.8 for r in di_ratios.values()
        )

        return {
            "selection_rates": selection_rates,
            "favored_group": favored_group,
            "di_ratios": di_ratios,
            "group_sizes": {
                group: len(scores) for group, scores in scores_by_group.items()
            },
            "four_fifths_threshold": 0.8,
            "violation": violation,
            "interpretation": self._interpret_di(
                di_ratios, violation, favored_rate
            ),
        }

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _interpret_delta(delta: float, threshold: float) -> str:
        """Human-readable interpretation of the counterfactual delta."""
        if abs(delta) < threshold:
            return (
                "No significant bias detected. The matching score is not "
                f"significantly affected by sensitive proxy variables "
                f"(Delta = {delta:.4f}, threshold = {threshold})."
            )
        elif delta > 0:
            return (
                f"Potential bias detected. The original resume scored "
                f"{delta:.4f} higher than the anonymised version, suggesting "
                "the model may be relying on sensitive proxy variables "
                "(e.g. name, gender, location)."
            )
        else:
            return (
                "Potential reverse bias detected. The anonymised resume "
                f"scored {abs(delta):.4f} higher than the original, "
                "which may indicate over-correction or unusual text patterns."
            )

    @staticmethod
    def _interpret_di(di_ratios: dict, violation: bool, favored_rate: float) -> str:
        """Human-readable interpretation of disparate impact."""
        if favored_rate == 0:
            return (
                "No candidates met the selection threshold in any group; "
                "the 4/5th Rule cannot distinguish selection rates."
            )
        if not violation:
            return (
                "No disparate impact violation detected. All groups pass "
                "the 4/5th Rule (ratio >= 0.80)."
            )
        failing = [g for g, r in di_ratios.items() if r < 0.8]
        return (
            "4/5th Rule violation detected. The following groups have a "
            f"selection rate below 80% of the favoured group: "
            f"{', '.join(failing)}."
        )
