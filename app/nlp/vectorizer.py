from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy
import pickle
import os
from pathlib import Path

# Load spaCy model for BERT-like embeddings
try:
    nlp = spacy.load("en_core_web_md")
except:
    # If model not found, download it
    os.system("python -m spacy download en_core_web_md")
    nlp = spacy.load("en_core_web_md")

class ResumeVectorizer:
    def __init__(self, method="tfidf"):
        self.method = method
        self.model_path = Path("app/nlp/models")
        self.model_path.mkdir(exist_ok=True)
        
        if method == "tfidf":
            self.tfidf_path = self.model_path / "tfidf_vectorizer.pkl"
            if self.tfidf_path.exists():
                with open(self.tfidf_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
            else:
                self.vectorizer = TfidfVectorizer(
                    max_features=5000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
    
    def _preprocess_text(self, text):
        """Preprocess text for vectorization"""
        # Basic preprocessing
        text = text.lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def vectorize_resume(self, resume_data):
        """Vectorize resume data"""
        # Combine all relevant text fields
        text_fields = [
            resume_data.get('full_text', ''),
            ' '.join(resume_data.get('skills', [])),
            ' '.join([edu.get('text', '') for edu in resume_data.get('education', [])]),
            ' '.join([exp.get('text', '') for exp in resume_data.get('experience', [])])
        ]
        
        text = ' '.join(text_fields)
        text = self._preprocess_text(text)
        
        if self.method == "tfidf":
            # If the vectorizer is not fitted yet, fit it
            if not hasattr(self.vectorizer, 'vocabulary_'):
                self.vectorizer.fit([text])
                with open(self.tfidf_path, 'wb') as f:
                    pickle.dump(self.vectorizer, f)
            
            # Transform the text to a vector
            vector = self.vectorizer.transform([text]).toarray()[0]
            return vector.tolist()  # Convert to list for MongoDB storage
        
        elif self.method == "spacy":
            # Use spaCy's word vectors
            doc = nlp(text)
            if doc.vector.any():  # Check if vector is not all zeros
                return doc.vector.tolist()
            else:
                return [0.0] * 300  # Default vector size for spaCy
    
    def vectorize_job_description(self, jd_data):
        """Vectorize job description data"""
        # Same process as resume vectorization
        return self.vectorize_resume(jd_data)
    
    def compute_similarity(self, resume_vector, jd_vector):
        """Compute cosine similarity between resume and job description vectors"""
        if not resume_vector or not jd_vector:
            return 0.0
        
        # Convert to numpy arrays
        resume_vector = np.array(resume_vector).reshape(1, -1)
        jd_vector = np.array(jd_vector).reshape(1, -1)
        
        # Compute cosine similarity
        similarity = cosine_similarity(resume_vector, jd_vector)[0][0]
        return float(similarity)
    
    def get_matching_skills(self, resume_skills, jd_skills):
        """Get matching and missing skills"""
        resume_skills_lower = [skill.lower() for skill in resume_skills]
        jd_skills_lower = [skill.lower() for skill in jd_skills]
        
        matching_skills = []
        missing_skills = []
        
        for skill in jd_skills:
            if skill.lower() in resume_skills_lower:
                matching_skills.append(skill)
            else:
                missing_skills.append(skill)
        
        return matching_skills, missing_skills
