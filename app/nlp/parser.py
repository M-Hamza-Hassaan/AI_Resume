import spacy
import re
import os
import docx
import PyPDF2
from pathlib import Path

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # If model not found, download it
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(docx_path):
    """Extract text from DOCX file"""
    doc = docx.Document(docx_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_file(file_path):
    """Extract text from PDF or DOCX file"""
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == '.docx':
        return extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def extract_email(text):
    """Extract email from text using regex"""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    return emails[0] if emails else None

def extract_phone(text):
    """Extract phone number from text using regex"""
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}'
    phones = re.findall(phone_pattern, text)
    # Flatten the first match
    if phones:
        return ''.join(phones[0])
    return None


def extract_skills(doc):
    """Extract skills from spaCy doc"""
    # Common skill-related keywords
    skill_patterns = [
        "experience with", "familiar with", "knowledge of", "skilled in",
        "proficient in", "expertise in", "competent in", "trained in"
    ]
    
    skills = set()
    
    # Extract skills based on patterns
    for pattern in skill_patterns:
        pattern_matches = re.finditer(pattern + r"\s([\w\s,]+)", doc.text, re.IGNORECASE)
        for match in pattern_matches:
            skill_text = match.group(1).strip()
            # Split by commas and 'and'
            for skill in re.split(r',|\sand\s', skill_text):
                if skill.strip():
                    skills.add(skill.strip())
    
    # Extract technical skills (nouns following technical adjectives)
    tech_adjectives = ["technical", "programming", "software", "hardware", "database", "web", "mobile"]
    for token in doc:
        if token.text.lower() in tech_adjectives and token.head.pos_ == "NOUN":
            skills.add(token.head.text)
    
    # Extract known programming languages, frameworks, etc.
    tech_skills = [
        "Python", "Java", "JavaScript", "C++", "C#", "Ruby", "PHP", "Swift",
        "React", "Angular", "Vue", "Django", "Flask", "Spring", "Node.js",
        "SQL", "MongoDB", "PostgreSQL", "MySQL", "Oracle", "AWS", "Azure",
        "Docker", "Kubernetes", "Git", "TensorFlow", "PyTorch", "scikit-learn"
    ]
    
    for skill in tech_skills:
        if re.search(r'\b' + re.escape(skill) + r'\b', doc.text, re.IGNORECASE):
            skills.add(skill)
    
    return list(skills)

def extract_education(doc):
    """Extract education information from spaCy doc"""
    education = []
    
    # Common education keywords
    edu_keywords = [
        "Bachelor", "Master", "PhD", "Doctorate", "BSc", "MSc", "BA", "MA",
        "degree", "university", "college", "school", "institute"
    ]
    
    # Find sentences containing education keywords
    for sent in doc.sents:
        if any(keyword.lower() in sent.text.lower() for keyword in edu_keywords):
            # Extract entities that might be institutions
            institutions = [ent.text for ent in sent.ents if ent.label_ in ["ORG", "GPE"]]
            
            # Extract years (could be graduation years)
            years = re.findall(r'\b(19|20)\d{2}\b', sent.text)
            
            if institutions or years:
                education.append({
                    "text": sent.text.strip(),
                    "institutions": institutions,
                    "years": years
                })
    
    return education

def extract_experience(doc):
    """Extract work experience from spaCy doc"""
    experience = []
    
    # Common work experience keywords
    work_keywords = [
        "work", "experience", "job", "position", "role", "employment",
        "worked", "employed", "hired", "manager", "director", "engineer",
        "developer", "analyst", "consultant", "intern"
    ]
    
    # Find sentences containing work keywords
    for sent in doc.sents:
        if any(keyword.lower() in sent.text.lower() for keyword in work_keywords):
            # Extract organizations
            organizations = [ent.text for ent in sent.ents if ent.label_ == "ORG"]
            
            # Extract years
            years = re.findall(r'\b(19|20)\d{2}\b', sent.text)
            
            # Extract job titles (capitalized phrases)
            job_titles = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', sent.text)
            
            if organizations or years or job_titles:
                experience.append({
                    "text": sent.text.strip(),
                    "organizations": organizations,
                    "years": years,
                    "job_titles": job_titles
                })
    
    return experience

def parse_resume(file_path):
    """Parse resume file and extract structured information"""
    # Extract text from file
    text = extract_text_from_file(file_path)
    
    # Process with spaCy
    doc = nlp(text)
    
    # Extract information
    email = extract_email(text)
    phone = extract_phone(text)
    skills = extract_skills(doc)
    education = extract_education(doc)
    experience = extract_experience(doc)
    
    # Return structured data
    return {
        "email": email,
        "phone": phone,
        "skills": skills,
        "education": education,
        "experience": experience,
        "full_text": text
    }

def parse_job_description(text):
    """Parse job description text and extract structured information"""
    # Process with spaCy
    doc = nlp(text)
    
    # Extract skills
    skills = extract_skills(doc)
    
    # Extract education requirements
    education = extract_education(doc)
    
    # Extract experience requirements
    experience = extract_experience(doc)
    
    # Return structured data
    return {
        "skills": skills,
        "education": education,
        "experience": experience,
        "full_text": text
    }
