import spacy
import re
import logging
import docx
import PyPDF2
from pathlib import Path

logger = logging.getLogger(__name__)


try:
    nlp = spacy.load("en_core_web_sm")
except OSError as error:
    raise RuntimeError(
        "spaCy model en_core_web_sm is required. Install it with: "
        "python -m spacy download en_core_web_sm"
    ) from error

_nlp_ner = None
_custom_model_path = Path(__file__).parent / "models" / "spacy_ner"

if _custom_model_path.exists():
    try:
        _nlp_ner = spacy.load(str(_custom_model_path))
        logger.info("Loaded custom NER model from %s", _custom_model_path)
    except Exception as e:
        logger.warning("Failed to load custom NER model: %s", e)
        _nlp_ner = None
else:
    logger.info(
        "Custom NER model not found at %s. "
        "Using en_core_web_sm default entities. "
        "Run: python -m app.nlp.train_ner to train.",
        _custom_model_path,
    )


def get_ner_entities(text: str) -> dict:
    """
    Extract resume-specific entities using the custom NER model if available.

    Returns a dict mapping entity labels to lists of entity text values:
        {
            "Name": ["John Doe"],
            "Skills": ["Python", "Flask", "SQL"],
            "Designation": ["Software Engineer"],
            "Companies worked at": ["Microsoft", "Google"],
            "College Name": ["MIT", "Stanford"],
            "Degree": ["B.Sc. Computer Science"],
            "Graduation Year": ["2020"],
            "Email Address": ["john@example.com"],
            "Location": ["London"],
        }
    """
    entities: dict = {}

    if _nlp_ner is not None:
        # Use custom-trained model
        ner_doc = _nlp_ner(text)
        for ent in ner_doc.ents:
            label = ent.label_
            if label not in entities:
                entities[label] = []
            # Avoid duplicates
            if ent.text.strip() not in entities[label]:
                entities[label].append(ent.text.strip())
    else:
        # Fallback: map en_core_web_sm labels to resume labels
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON" and "Name" not in entities:
                entities["Name"] = [ent.text.strip()]
            elif ent.label_ == "ORG":
                entities.setdefault("Companies worked at", [])
                if ent.text.strip() not in entities["Companies worked at"]:
                    entities["Companies worked at"].append(ent.text.strip())
            elif ent.label_ == "GPE":
                entities.setdefault("Location", [])
                if ent.text.strip() not in entities["Location"]:
                    entities["Location"].append(ent.text.strip())

    return entities


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file."""
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += (page.extract_text() or "") + "\n"
    return text


def extract_text_from_docx(docx_path):
    """Extract text from DOCX file."""
    doc = docx.Document(docx_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def extract_text_from_file(file_path):
    """Extract text from PDF or DOCX file."""
    file_extension = Path(file_path).suffix.lower()

    if file_extension == ".pdf":
        return extract_text_from_pdf(file_path)
    elif file_extension == ".docx":
        return extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")



def extract_email(text):
    """Extract email from text using regex."""
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    emails = re.findall(email_pattern, text)
    return emails[0] if emails else None


def extract_phone(text):
    phone_pattern = (
        r"(?<!\d)"                          # not preceded by a digit
        r"(?:\+\d{1,3}[\s\-.]*)?"        # optional country code (+92, +1, +966)
        r"(?:\(?\d{2,4}\)?[\s\-.]*)?"   # optional area code  (321), 021
        r"\d{3,5}"                           # first digit block
        r"(?:[\s\-.]*\d{2,5})+"           # one or more additional digit blocks
        r"(?!\d)"                            # not followed by a digit
    )

    matches = re.search(phone_pattern, text)
    return matches.group(0).strip() if matches else None



def extract_skills(doc, ner_entities=None):
    skills = set()

    # 1. Use NER-extracted skills if available
    if ner_entities and "Skills" in ner_entities:
        for skill in ner_entities["Skills"]:
            # NER may return a long block of skills — split by common delimiters
            for s in re.split(r"[,\n•\-\|]", skill):
                s = s.strip()
                if s and len(s) > 1:
                    # Remove parenthetical durations like "(Less than 1 year)"
                    s = re.sub(r"\s*\(.*?\)\s*", "", s).strip()
                    if s:
                        skills.add(s)

    # 2. Pattern-based extraction ("proficient in Python", "skilled in Java")
    skill_patterns = [
        "experience with", "familiar with", "knowledge of", "skilled in", "used", "skills",
        "proficient in", "expertise in", "competent in", "trained in", "worked with",
    ]
    for pattern in skill_patterns:
        pattern_matches = re.finditer(
            pattern + r"\s([\w\s,#.+\-]+)", doc.text, re.IGNORECASE
        )
        for match in pattern_matches:
            skill_text = match.group(1).strip()
            for skill in re.split(r",|\sand\s", skill_text):
                if skill.strip():
                    skills.add(skill.strip())

    # 3. Hardcoded tech skills list (case-insensitive word boundary match)
    tech_skills = [
        "Python", "Java", "JavaScript", "C++", "C#", "Ruby", "PHP", "Swift",
        "React.js", "Angular", "Vue", "Django", "Flask", "Spring", "Node.js",
        "SQL", "MongoDB", "PostgreSQL", "MySQL", "Oracle", "AWS", "Azure",
        "Docker", "Kubernetes", "Git", "TensorFlow", "PyTorch", "scikit-learn",
        "HTML", "CSS", "Linux", "Windows", "MacOS", "Jenkins", "Agile",
        "Scrum", "REST", "GraphQL", "TypeScript", "Go", "Rust", "Kotlin",
        "Flutter", "React Native", "Tableau", "Power BI", "Spark", "Hadoop",
        "Kafka", "Redis", "Elasticsearch", "Microservices", "CI/CD", "DevOps",
        "NLP", "Machine Learning", "Deep Learning", "Data Science",
        "Computer Vision", "OpenCV", "Pandas", "NumPy", "Matplotlib",
    ]
    for skill in tech_skills:
        if re.search(r"\b" + re.escape(skill) + r"\b", doc.text, re.IGNORECASE):
            skills.add(skill)

    return sorted(skills, key=str.casefold)


def _iter_segments(doc):
    """Yield resume lines (or sentence spans) with entities local to each span."""
    text = doc.text
    segments = []
    if "\n" in text or "\r" in text:
        offset = 0
        for line in text.splitlines(keepends=True):
            line_end = offset + len(line)
            content_start = offset + len(line) - len(line.lstrip())
            content_end = offset + len(line.rstrip())
            if content_start < content_end:
                segments.append((text[content_start:content_end], content_start, content_end))
            offset = line_end
    else:
        segments = [
            (sentence.text.strip(), sentence.start_char, sentence.end_char)
            for sentence in doc.sents
            if sentence.text.strip()
        ]

    entities = list(doc.ents)
    for segment, start, end in segments:
        segment_entities = [
            entity
            for entity in entities
            if entity.start_char >= start and entity.end_char <= end
        ]
        yield segment, segment_entities


def extract_education(doc, ner_entities=None):

    education = []

    edu_keywords = [
        "Bachelor", "Master", "PhD", "Doctorate", "BSc", "MSc", "BA", "MA",
        "degree", "university", "college", "school", "institute",
        "B.Tech", "M.Tech", "MCA", "BCA", "MBA", "B.S", "M.S", "BS", "MS",
        "diploma",
    ]

    # Use NER entities for institutions and years — use exact dataset label names
    ner_colleges = ner_entities.get("College Name", []) if ner_entities else []
    ner_degrees = ner_entities.get("Degree", []) if ner_entities else []
    ner_years = ner_entities.get("Graduation Year", []) if ner_entities else []

    # Find sentences containing education keywords
    for segment, segment_entities in _iter_segments(doc):
        sent_lower = segment.lower()
        if any(keyword.lower() in sent_lower for keyword in edu_keywords):
            # Try NER entities first, fall back to en_core_web_sm
            institutions = []
            for college in ner_colleges:
                if college.lower() in sent_lower:
                    institutions.append(college)
            if not institutions:
                institutions = [
                    ent.text for ent in segment_entities
                    if ent.label_ in ["ORG", "GPE"]
                ]

            # Years from NER or regex
            years = []
            for year in ner_years:
                if year in segment:
                    years.append(year)
            if not years:
                years = re.findall(r"\b(?:19|20)\d{2}\b", segment)

            if institutions or years:
                education.append({
                    "text": segment.strip(),
                    "institutions": institutions,
                    "years": years,
                })

    # If NER detected degrees/colleges but no sentences matched, add them directly
    if ner_degrees and not education:
        education.append({
            "text": ", ".join(ner_degrees),
            "institutions": ner_colleges,
            "years": ner_years,
        })

    return education


def extract_experience(doc, ner_entities=None):
    """
    Extract work experience from resume text.

    Uses the custom NER model's "Designation" and "Companies worked at"
    entities if available, supplemented by keyword matching.
    """
    experience = []

    work_keywords = [
        "work", "experience", "job", "position", "role", "employment",
        "worked", "employed", "hired", "manager", "director", "engineer",
        "developer", "analyst", "consultant", "intern",
    ]

    # NER entities
    ner_companies = ner_entities.get("Companies worked at", []) if ner_entities else []
    ner_designations = ner_entities.get("Designation", []) if ner_entities else []

    for segment, segment_entities in _iter_segments(doc):
        sent_lower = segment.lower()
        if any(keyword.lower() in sent_lower for keyword in work_keywords):
            # Organizations from NER or en_core_web_sm
            organizations = []
            for company in ner_companies:
                if company.lower() in sent_lower:
                    organizations.append(company)
            if not organizations:
                organizations = [
                    ent.text for ent in segment_entities if ent.label_ == "ORG"
                ]

            # Years
            years = re.findall(r"\b(?:19|20)\d{2}\b", segment)
            duration = re.search(
                r"\b(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b",
                segment,
                re.IGNORECASE,
            )

            # Job titles from NER or regex
            job_titles = []
            for designation in ner_designations:
                if designation.lower() in sent_lower:
                    job_titles.append(designation)
            if not job_titles:
                job_titles = re.findall(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", segment)

            if organizations or years or job_titles or duration:
                experience.append({
                    "text": segment.strip(),
                    "organizations": organizations,
                    "years": years,
                    "job_titles": job_titles,
                    "duration_years": (
                        float(duration.group(1)) if duration else None
                    ),
                })

    return experience


def extract_name(ner_entities=None, text=""):
    """Extract candidate name from NER entities or text."""
    if ner_entities and "Name" in ner_entities and ner_entities["Name"]:
        return ner_entities["Name"][0]
    # Fallback: first non-empty line that looks like a name
    lines = text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line and len(line.split()) >= 2 and len(line.split()) <= 5:
            # Skip lines with numbers or special chars
            if not re.search(r"[@\d]", line):
                return line
    return None


def extract_years_of_experience(text, ner_entities=None):
    """Extract an explicitly stated total experience duration, if available."""
    patterns = []
    if ner_entities:
        patterns.extend(ner_entities.get("Years of Experience", []))
    patterns.append(text)

    match_pattern = re.compile(
        r"\b(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b",
        re.IGNORECASE,
    )
    for candidate in patterns:
        match = match_pattern.search(candidate or "")
        if match:
            return float(match.group(1))
    return None


def parse_resume(file_path):
    
    # Extract raw text
    text = extract_text_from_file(file_path)

    # Process with base model (for sentences, POS, dependency)
    doc = nlp(text)

    # Extract entities with custom NER model (if available)
    ner_entities = get_ner_entities(text)

    # Extract information
    name = extract_name(ner_entities, text)
    email = extract_email(text)
    phone = extract_phone(text)
    years_of_experience = extract_years_of_experience(text, ner_entities)
    skills = extract_skills(doc, ner_entities)
    education = extract_education(doc, ner_entities)
    experience = extract_experience(doc, ner_entities)

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "years_of_experience": years_of_experience,
        "skills": skills,
        "education": education,
        "experience": experience,
        "full_text": text,
    }


def parse_job_description(text):
    
    doc = nlp(text)
    ner_entities = get_ner_entities(text)

    skills = extract_skills(doc, ner_entities)
    education = extract_education(doc, ner_entities)
    experience = extract_experience(doc, ner_entities)

    return {
        "skills": skills,
        "education": education,
        "experience": experience,
        "full_text": text,
    }
