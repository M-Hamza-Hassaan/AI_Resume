from app.models.resume import Resume
from app.models.job_description import JobDescription
from app.nlp.vectorizer import ResumeVectorizer
from bson.objectid import ObjectId
from app import mongo

class ResumeRanker:
    def __init__(self, vectorizer_method="tfidf"):
        self.vectorizer = ResumeVectorizer(method=vectorizer_method)
    
    def rank_resumes_for_job(self, job_description_id):
        """Rank all resumes for a specific job description"""
        # Get job description
        jd = JobDescription.get_by_id(job_description_id)
        if not jd:
            return []
        
        # Get all resumes
        resumes = Resume.get_all()
        
        # Calculate similarity scores
        ranked_resumes = []
        for resume in resumes:
            # Skip if no vector
            if not resume.vector or not jd.vector:
                continue
            
            # Calculate similarity score
            similarity = self.vectorizer.compute_similarity(resume.vector, jd.vector)
            
            # Get matching and missing skills
            matching_skills, missing_skills = self.vectorizer.get_matching_skills(
                resume.parsed_data.get('skills', []),
                jd.parsed_data.get('skills', [])
            )
            
            # Get user info
            user_data = mongo.db.users.find_one({"_id": ObjectId(resume.user_id)})
            user_name = user_data.get('name', 'Unknown') if user_data else 'Unknown'
            
            ranked_resumes.append({
                'resume_id': resume.id,
                'user_id': resume.user_id,
                'user_name': user_name,
                'similarity_score': similarity,
                'matching_skills': matching_skills,
                'missing_skills': missing_skills,
                'skills_match_percentage': len(matching_skills) / len(jd.parsed_data.get('skills', [])) * 100 if jd.parsed_data.get('skills', []) else 0
            })
        
        # Sort by similarity score (descending)
        ranked_resumes.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return ranked_resumes
    
    def get_resume_insights(self, resume_id, job_description_id):
        """Get detailed insights for a specific resume and job description"""
        # Get resume and job description
        resume = Resume.get_by_id(resume_id)
        jd = JobDescription.get_by_id(job_description_id)
        
        if not resume or not jd:
            return None
        
        # Calculate similarity score
        similarity = self.vectorizer.compute_similarity(resume.vector, jd.vector)
        
        # Get matching and missing skills
        matching_skills, missing_skills = self.vectorizer.get_matching_skills(
            resume.parsed_data.get('skills', []),
            jd.parsed_data.get('skills', [])
        )
        
        # Calculate skill match percentage
        skills_match_percentage = len(matching_skills) / len(jd.parsed_data.get('skills', [])) * 100 if jd.parsed_data.get('skills', []) else 0
        
        # Get education match
        education_match = self._evaluate_education_match(
            resume.parsed_data.get('education', []),
            jd.parsed_data.get('education', [])
        )
        
        # Get experience match
        experience_match = self._evaluate_experience_match(
            resume.parsed_data.get('experience', []),
            jd.parsed_data.get('experience', [])
        )
        
        return {
            'similarity_score': similarity,
            'matching_skills': matching_skills,
            'missing_skills': missing_skills,
            'skills_match_percentage': skills_match_percentage,
            'education_match': education_match,
            'experience_match': experience_match
        }
    
    def _evaluate_education_match(self, resume_education, jd_education):
        """Evaluate how well the resume's education matches job requirements"""
        # This is a simplified implementation
        # In a real system, you would use more sophisticated NLP techniques
        
        if not jd_education:
            return {'match': True, 'score': 1.0, 'details': 'No specific education requirements'}
        
        if not resume_education:
            return {'match': False, 'score': 0.0, 'details': 'No education information provided'}
        
        # Check for degree keywords in both
        resume_edu_text = ' '.join([edu.get('text', '') for edu in resume_education]).lower()
        jd_edu_text = ' '.join([edu.get('text', '') for edu in jd_education]).lower()
        
        degree_keywords = ['bachelor', 'master', 'phd', 'doctorate', 'bsc', 'msc', 'ba', 'ma']
        
        for keyword in degree_keywords:
            if keyword in jd_edu_text and keyword in resume_edu_text:
                return {'match': True, 'score': 1.0, 'details': f'Found matching {keyword} degree'}
        
        # If no direct match, give partial score
        return {'match': False, 'score': 0.5, 'details': 'Education requirements partially met'}
    
    def _evaluate_experience_match(self, resume_experience, jd_experience):
        """Evaluate how well the resume's experience matches job requirements"""
        # This is a simplified implementation
        
        if not jd_experience:
            return {'match': True, 'score': 1.0, 'details': 'No specific experience requirements'}
        
        if not resume_experience:
            return {'match': False, 'score': 0.0, 'details': 'No experience information provided'}
        
        # Extract years of experience from job description
        import re
        
        jd_exp_text = ' '.join([exp.get('text', '') for exp in jd_experience])
        years_pattern = r'(\d+)[\+]?\s+years?'
        years_match = re.search(years_pattern, jd_exp_text)
        
        if years_match:
            required_years = int(years_match.group(1))
            
            # Estimate years from resume (very simplified)
            resume_years = len(resume_experience)
            
            if resume_years >= required_years:
                return {'match': True, 'score': 1.0, 'details': f'Meets {required_years}+ years requirement'}
            else:
                score = resume_years / required_years
                return {'match': False, 'score': score, 'details': f'Has {resume_years} years, needs {required_years}'}
        
        # If no years requirement found, check for keyword matches
        return {'match': True, 'score': 0.7, 'details': 'Experience appears relevant'}
