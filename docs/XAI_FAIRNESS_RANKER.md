# airesume — Technical Documentation: XAI, Fairness, Dual Ranker & NER Training

This document provides detailed technical documentation for the four modules
added to airesume as specified in the MSc research blueprint (`readme.md`).

---

## Table of Contents

1. [Module Overview](#module-overview)
2. [ranker.py — Dual Matching Engine](#1-rankerpy--dual-matching-engine-rq2)
3. [xai.py — Explainable AI Engine](#2-xaisy--explainable-ai-engine)
4. [fairness.py — Bias Mitigation & Audit](#3-fairnesspy--bias-mitigation--audit-rq3)
5. [train_ner.py — Custom NER Model Training](#4-train_nerpy--custom-ner-model-training-rq1)
6. [Integration Architecture](#integration-architecture)
7. [Data Flow](#data-flow)
8. [Template Integration](#template-integration)

---

## Module Overview

| Module | File | RQ | Responsibility |
|--------|------|----|---------------|
| Dual Ranker | `app/nlp/ranker.py` | RQ2 | TF-IDF + SBERT dual matching with section-weighted scoring |
| Explainable AI | `app/nlp/xai.py` | — | Skill gap analysis + SHAP-style score breakdown |
| Fairness | `app/nlp/fairness.py` | RQ3 | Text anonymisation + counterfactual bias auditing |
| NER Training | `app/nlp/train_ner.py` | RQ1 | Train custom spaCy v3 NER on Dataturks resume dataset |

---

## 1. `ranker.py` — Dual Matching Engine (RQ2)

### 1.1 Architecture

The ranker implements the blueprint's dual-model matching pipeline:

```
Resume Text ─────┬──────────────────────────┬───────────────────────┐
                 ▼                           ▼                       ▼
         TFIDFMatcher                SBERTMatcher            Section Scoring
    (stored TF-IDF vectors)    (all-MiniLM-L6-v2 384-dim)   (skills/exp/edu)
                 │                           │                       │
                 └───────────┬───────────────┘                       │
                             ▼                                       │
                    Text Similarity                                   │
                   (TF-IDF or avg)                                    │
                             │                                         │
                             └──────────────┬──────────────────────────┘
                                            ▼
                                   Combined Score
                            (60% text sim + 40% section)
                                            │
                                            ▼
                                    XAI Report
                              (skill gap + SHAP)
```

### 1.2 Classes

#### `TFIDFMatcher`

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `compute_similarity(resume_vector, jd_vector)` | Two TF-IDF vectors (lists from MongoDB) | `float [0,1]` | Cosine similarity from pre-computed stored vectors |
| `compute_text_similarity(resume_text, jd_text)` | Two raw text strings | `float [0,1]` | Builds TF-IDF on the fly from raw text (used in counterfactual testing) |

**Formula:**

```
Sim_TFIDF(JD, R) = Cosine(Vector_JD, Vector_R)
```

#### `SBERTMatcher`

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `encode(text)` | Raw text string | `np.ndarray (384,)` | 384-dim Sentence-BERT embedding (truncated to 400 words) |
| `compute_similarity(resume_text, jd_text)` | Two raw text strings | `float [0,1]` | Encodes both texts and computes cosine similarity |
| `encode_batch(texts)` | List of text strings | `np.ndarray (n, 384)` | Batch encoding for efficiency |

**Formula:**

```
E_JD = SBERT(Job Description)
E_R  = SBERT(Resume)
Sim_SBERT(JD, R) = Cosine(E_JD, E_R)
```

**Model:** `all-MiniLM-L6-v2` (384-dimensional, ~90MB, fast on CPU)

#### `ResumeRanker`

The unified ranker that orchestrates both matchers with XAI and fairness.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vectorizer_method` | `str` | `"tfidf"` | Baseline vectoriser method |
| `use_sbert` | `bool` | `False` | Enable SBERT semantic matching |
| `toggle_anonymization` | `bool` | `False` | Mask sensitive proxies before matching |

| Method | Returns | Description |
|--------|---------|-------------|
| `rank_resumes_for_job(jd_id)` | `list[dict]` | Ranked list of all resumes with dual scores, XAI reports, skill coverage |
| `get_resume_insights(resume_id, jd_id)` | `dict` | Full insights dict with dual scores, section scores, XAI report, legacy match dicts |
| `compute_text_similarity(resume_text, jd_text)` | `float` | Raw text similarity for fairness testing |

### 1.3 Scoring Formula

```
# Text similarity (blend if both models available)
if sbert_score is not None:
    text_sim = (tfidf_score + sbert_score) / 2.0
else:
    text_sim = tfidf_score

# Section-weighted score
section_score = skills_score * 0.50 + experience_score * 0.30 + education_score * 0.20

# Final combined score
combined_score = 0.60 * text_sim + 0.40 * section_score
```

### 1.4 Ranked Resume Dict Structure

Each item returned by `rank_resumes_for_job()`:

```python
{
    "resume_id": "...",
    "user_id": "...",
    "user_name": "Jane Doe",
    "similarity_score": 0.78,        # combined final score
    "tfidf_score": 0.72,             # TF-IDF cosine similarity
    "sbert_score": 0.84,             # SBERT cosine similarity (or None)
    "combined_score": 0.78,          # same as similarity_score
    "matching_skills": ["Python", "Flask"],
    "missing_skills": ["Docker"],
    "skills_match_percentage": 66.7,
    "skills_score": 0.67,           # normalized [0,1]
    "experience_score": 0.80,
    "education_score": 0.90,
    "xai_report": { ... }            # full XAI report (see below)
}
```

---

## 2. `xai.py` — Explainable AI Engine

### 2.1 Purpose

Makes the ranking transparent by decomposing the final score into
per-factor contributions and providing skill gap analysis with visual badges.

### 2.2 Class: `XAIExplainer`

#### Section Weights

| Section | Weight | Description |
|---------|--------|-------------|
| Skills | 50% | Skills match coverage |
| Experience | 30% | Years and relevance |
| Education | 20% | Degree level match |

#### Methods

##### `analyze_skill_coverage(resume_skills, jd_skills)`

Compares skills case-insensitively and returns:

```python
{
    "matched_skills": ["Python", "Flask", "SQL"],
    "missing_skills": ["Docker", "Kubernetes"],
    "additional_skills": ["Java"],     # in resume but not JD
    "coverage_percentage": 60.0,
    "coverage_level": "moderate",      # "high" >= 75%, "moderate" >= 40%, "low" < 40%
    "total_jd_skills": 5,
    "matched_count": 3,
    "missing_count": 2,
}
```

##### `generate_feature_breakdown(tfidf_score, sbert_score, skills_score, experience_score, education_score)`

Returns a feature contribution table:

```python
{
    "features": [
        {
            "name": "Skills Match",
            "value": 0.85,           # raw score
            "weight": 0.50,         # section weight
            "contribution": 0.425,  # value * weight
            "icon": "fa-check-circle",
            "color": "success"      # bootstrap color class
        },
        {
            "name": "Experience Match",
            "value": 0.60,
            "weight": 0.30,
            "contribution": 0.18,
            "icon": "fa-briefcase",
            "color": "warning"
        },
        {
            "name": "Education Match",
            "value": 0.90,
            "weight": 0.20,
            "contribution": 0.18,
            "icon": "fa-graduation-cap",
            "color": "success"
        },
        # Informational (not part of weighted total):
        {
            "name": "TF-IDF Similarity",
            "value": 0.72,
            "weight": 0.0,
            "contribution": 0.0,
            "icon": "fa-search",
            "color": "info",
            "informational": True
        },
        {
            "name": "SBERT Similarity",
            "value": 0.84,
            "weight": 0.0,
            "contribution": 0.0,
            "icon": "fa-brain",
            "color": "info",
            "informational": True
        },
    ],
    "total_score": 0.785,
    "summary": "Overall match score: 78.5%. Strongest factor: Skills Match (42.5% contribution)."
}
```

##### `generate_shap_style_breakdown(feature_scores)`

Creates waterfall-style decomposition data:

```python
{
    "baseline": 0.783,        # mean of all feature scores
    "contributions": [
        {
            "name": "Skills",
            "score": 0.85,
            "baseline": 0.783,
            "delta": 0.067,     # score - baseline
            "cumulative": 0.850,
            "positive": True    # delta >= 0
        },
        # ... per feature
    ],
    "final_score": 0.783
}
```

##### `explain_ranking(resume_parsed_data, jd_parsed_data, ...)`

Full report combining all of the above:

```python
{
    "skill_coverage": { ... },
    "feature_breakdown": { ... },
    "shap_breakdown": { ... },
    "badges": [
        {"skill": "Python", "status": "matched", "class": "bg-success", "icon": "fa-check"},
        {"skill": "Docker", "status": "missing", "class": "bg-danger", "icon": "fa-times"},
        {"skill": "Java", "status": "additional", "class": "bg-info", "icon": "fa-plus"},
    ],
    "summary": "Overall match score: 78.5%..."
}
```

---

## 3. `fairness.py` — Bias Mitigation & Audit (RQ3)

### 3.1 Purpose

Prevents the matching model from relying on sensitive proxy variables
(name, gender, location, dates, college names) by:
1. Masking these variables before vectorisation
2. Testing whether masking changes the score (counterfactual test)
3. Testing selection rate fairness across groups (4/5th Rule)

### 3.2 Class: `TextAnonymizer`

#### Proxy Variable Mappings

| Original Entity | Detection Method | Replacement Token |
|-----------------|-------------------|-------------------|
| Person names | spaCy NER `PERSON` | `[CANDIDATE_NAME]` |
| Dates/years | Regex (4-digit years, month names) | `[GRADUATION_YEAR]` |
| Gender pronouns | Regex word match (he, she, Mr, Mrs...) | `[PRONOUN]` |
| College names | spaCy NER `ORG` + keyword detection | `[COLLEGE_NAME]` |
| Locations | spaCy NER `GPE` | `[LOCATION]` |

#### Methods

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `anonymize(text)` | Raw text string | Anonymised text | Masks all sensitive proxies |
| `get_mask_summary()` | — | `dict` | Summary of what was masked: `{proxy_type: [original_values]}` |

#### Usage

```python
anonymizer = TextAnonymizer(toggle_anonymization=True)
clean_text = anonymizer.anonymize(resume_text)
# "John Doe worked at Microsoft in London" -> "[CANDIDATE_NAME] worked at Microsoft in [LOCATION]"
```

### 3.3 Class: `FairnessAuditor`

#### `counterfactual_test(resume_text, jd_text, matching_fn=None)`

Creates an anonymised copy of the resume, re-scores it, and computes Delta:

```
Delta = Score(CV_original) - Score(CV_anonymised)
```

| Delta Value | Interpretation |
|-------------|---------------|
| `abs(Delta) < 0.05` | No significant bias detected |
| `Delta > 0.05` | Model relies on sensitive proxies (original scored higher) |
| `Delta < -0.05` | Reverse bias (anonymised scored higher) |

Returns:

```python
{
    "original_score": 0.78,
    "anonymised_score": 0.76,
    "delta": 0.02,
    "bias_detected": False,          # abs(delta) > 0.05
    "bias_threshold": 0.05,
    "mask_summary": {
        "NAME": ["John Doe"],
        "LOCATION": ["London"],
        "DATE": ["2020", "2021"],
    },
    "interpretation": "No significant bias detected..."
}
```

#### `disparate_impact_test(scores_by_group, selection_threshold=0.5)`

Implements the 4/5th Rule:

```
For each group: selection_rate = count(scores >= threshold) / total_scores
DI_ratio = group_rate / max_rate
Violation if any DI_ratio < 0.80
```

Returns:

```python
{
    "selection_rates": {"group_a": 0.60, "group_b": 0.40},
    "favored_group": "group_a",
    "di_ratios": {"group_a": 1.0, "group_b": 0.667},
    "four_fifths_threshold": 0.80,
    "violation": True,              # group_b ratio < 0.80
    "interpretation": "4/5th Rule violation detected..."
}
```

---

## 4. `train_ner.py` — Custom NER Model Training (RQ1)

### 4.1 Purpose

Trains a custom spaCy v3 NER model on the Dataturks resume dataset (220
annotated resumes) to extract resume-specific entities. Replaces the outdated
spaCy v2 training in `Entity-Recognition-In-Resumes-SpaCy/train.py`.

### 4.2 Entity Labels

| Label | Description | Example |
|-------|-------------|---------|
| `Name` | Candidate full name | "Abhishek Jha" |
| `Designation` | Job title / role | "Application Development Associate" |
| `Skills` | Technical skills | "Python, Flask, SQL" |
| `Companies worked at` | Employer names | "Accenture, Microsoft" |
| `College Name` | Educational institution | "B.v.b college of engineering" |
| `Degree` | Academic degree | "B.E in Information Science" |
| `Graduation Year` | Year of graduation | "2017" |
| `Email Address` | Email / indeed URL | "indeed.com/r/Abhishek-Jha" |
| `Location` | City / state | "Bengaluru, Karnataka" |
| `Years of Experience` | Experience duration | "3.5 years" |

### 4.3 Training Pipeline

```
Dataturks JSON (220 resumes)
        │
        ▼
convert_dataturks_to_examples()
  • Parse JSON lines (content + annotation)
  • Convert inclusive [start, end] → exclusive [start, end)
  • Filter: trim whitespace + punctuation, remove overlaps
        │
        ▼
_filter_entities()
  • Trim leading/trailing whitespace + punctuation (fixes E024)
  • Remove out-of-bounds offsets
  • Remove overlapping spans
        │
        ▼
save_docbin() → train.spacy, dev.spacy (spaCy v3 DocBin)
        │
        ▼
train_ner_model()
  • spacy.blank("en") + NER pipe
  • 30 iterations, compounding batch size (4→32)
  • Dropout = 0.2
  • 85/15 train/dev split
        │
        ▼
_evaluate() — Precision / Recall / F-score on dev set
        │
        ▼
Save best model → app/nlp/models/spacy_ner/
```

### 4.4 Key Functions

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `convert_dataturks_to_examples(json_path, nlp)` | JSON file path, spaCy nlp | `List[Example]` | Convert Dataturks JSON to spaCy v3 Example objects |
| `_filter_entities(text, entities)` | Text, entity list | filtered entity list | Trim whitespace/punctuation, remove overlaps |
| `save_docbin(examples, path)` | Examples, output path | `.spacy` file | Save as spaCy v3 DocBin binary |
| `train_ner_model(json_path, output_dir, ...)` | JSON path, output dir | trained `Language` | Full training pipeline |
| `_evaluate(nlp, dev_examples)` | Model, dev examples | `{p, r, f}` | Compute precision/recall/F-score |

### 4.5 Training Results

| Metric | Value |
|--------|-------|
| Training examples | 187 |
| Dev examples | 33 |
| Iterations completed | 26/30 (timeout) |
| Best F-score | 0.5533 (55.3%) |
| Best Precision | 0.6095 |
| Best Recall | 0.5066 |

### 4.6 CLI Usage

```bash
# Default paths (looks for JSON in parent workspace)
python -m app.nlp.train_ner

# Custom paths and iterations
python -m app.nlp.train_ner \
    --data "path/to/Entity_Recognition_in_Resumes.json" \
    --output "app/nlp/models/spacy_ner" \
    --iter 30 --split 0.85 --dropout 0.2
```

### 4.7 Parser Integration

`parser.py` loads the custom NER model at startup:

```python
# Custom NER model for resume-specific entities
_nlp_ner = None
_custom_model_path = Path(__file__).parent / "models" / "spacy_ner"
if _custom_model_path.exists():
    _nlp_ner = spacy.load(str(_custom_model_path))
```

The base `en_core_web_sm` is always loaded for sentence segmentation and POS
tagging. The custom NER model is used via `get_ner_entities(text)` which returns
a dict of label → [entity_text] for resume-specific labels. If the custom model
is not found, it falls back to mapping `en_core_web_sm` entities (PERSON → Name,
ORG → Companies, GPE → Location).

---

## Integration Architecture

### How the modules connect

```
recruiter.py (route)
    │
    ├── Creates ResumeRanker(use_sbert=True, toggle_anonymization=True)
    │       │
    │       ├── TFIDFMatcher.compute_similarity()      # from stored MongoDB vectors
    │       ├── SBERTMatcher.compute_similarity()      # from full_text, after anonymization
    │       ├── _compute_skills_score()                # matched/total ratio
    │       ├── _compute_experience_score()             # regex years extraction
    │       ├── _compute_education_score()              # degree keyword matching
    │       ├── _compute_combined_score()               # 60% text + 40% section
    │       │
    │       └── XAIExplainer.explain_ranking()
    │               ├── analyze_skill_coverage()        # skill gap with badges
    │               ├── generate_feature_breakdown()    # contribution table
    │               └── generate_shap_style_breakdown() # waterfall data
    │
    └── FairnessAuditor.counterfactual_test()
            ├── TextAnonymizer.anonymize()              # mask sensitive proxies
            └── ranker.compute_text_similarity()         # re-score anonymized text
```

### Route → Template Data Flow

**`/recruiter/rank-resumes/<jd_id>`** passes to `ranked_resumes.html`:
- `ranked_resumes`: list of dicts with `similarity_score`, `tfidf_score`, `sbert_score`, `skills_match_percentage`, `xai_report`
- `sbert_enabled`: boolean for conditional dual-model columns
- `jd`: job description object with `parsed_data.skills`

**`/recruiter/resume-insights/<jd_id>/<resume_id>`** passes to `resume_insights.html`:
- `insights`: dict with `similarity_score`, `tfidf_score`, `sbert_score`, `skills_score`, `experience_score`, `education_score`, `education_match`, `experience_match`, `xai_report`
- `fairness_result`: dict with `original_score`, `anonymised_score`, `delta`, `bias_detected`, `mask_summary`, `interpretation`
- `sbert_enabled`: boolean
- `resume`, `jd`: model objects

---

## Template Integration

### `ranked_resumes.html`

New table columns:
- **Combined Score** — progress bar with combined match percentage
- **TF-IDF** — badge showing TF-IDF similarity (conditional on SBERT)
- **SBERT** — badge showing SBERT similarity (conditional on SBERT)
- **Skills Match** — progress bar
- **XAI Summary** — truncated text summary from XAI report
- **Actions** — link to XAI Insights page

Top candidate skill badges section showing matched (green), missing (red), and additional (blue) badges from `xai_report.badges`.

### `resume_insights.html`

New sections:
1. **Dual Model Scores** — TF-IDF and SBERT progress bars side by side
2. **Section Scores** — Skills (50%), Experience (30%), Education (20%) with weights
3. **XAI Feature Contribution Breakdown** — table with per-factor: score, weight, contribution, color-coded
4. **SHAP-Style Score Decomposition** — waterfall showing baseline, deltas, and final score
5. **Skill Gap Analysis** — coverage percentage, level badge, matched/missing/additional skill badges
6. **Fairness Audit — Counterfactual Bias Test** — score comparison table, delta, bias status, masked proxy variables summary, interpretation alert
7. **Education & Experience Details** — existing parsed data cards

---

## Graceful Degradation

| Scenario | Behaviour |
|----------|-----------|
| `sentence-transformers` not installed | `SBERT_AVAILABLE = False`, ranker falls back to TF-IDF only, SBERT columns hidden in templates |
| `shap` not installed | SHAP-style breakdown uses custom implementation (no shap import needed) |
| spaCy model not downloaded | Auto-downloads `en_core_web_sm` on first use |
| MongoDB not running | Flask app crashes on startup (expected) |
| Resume has no `full_text` | SBERT matching returns 0.0, fairness test skipped |
| JD has no skills detected | Skills score = 0.0, coverage = 0% |
