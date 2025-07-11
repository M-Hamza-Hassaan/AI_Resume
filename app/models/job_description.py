from app import mongo
from bson.objectid import ObjectId
import datetime

class JobDescription:
    def __init__(self, jd_data=None):
        if jd_data:
            self.id = str(jd_data.get('_id'))
            self.recruiter_id = jd_data.get('recruiter_id')
            self.title = jd_data.get('title')
            self.company = jd_data.get('company')
            self.description = jd_data.get('description')
            self.upload_date = jd_data.get('upload_date')
            self.parsed_data = jd_data.get('parsed_data', {})
            self.vector = jd_data.get('vector')
        else:
            self.id = None
            self.recruiter_id = None
            self.title = None
            self.company = None
            self.description = None
            self.upload_date = datetime.datetime.now()
            self.parsed_data = {}
            self.vector = None
    
    def save(self):
        jd_data = {
            'recruiter_id': self.recruiter_id,
            'title': self.title,
            'company': self.company,
            'description': self.description,
            'upload_date': self.upload_date,
            'parsed_data': self.parsed_data,
            'vector': self.vector
        }
        
        if self.id:
            mongo.db.job_descriptions.update_one(
                {'_id': ObjectId(self.id)},
                {'$set': jd_data}
            )
        else:
            result = mongo.db.job_descriptions.insert_one(jd_data)
            self.id = str(result.inserted_id)
            
            # Add job description ID to recruiter's list
            mongo.db.users.update_one(
                {'_id': ObjectId(self.recruiter_id)},
                {'$push': {'job_descriptions': self.id}}
            )
        
        return self.id
    
    @staticmethod
    def get_by_id(jd_id):
        jd_data = mongo.db.job_descriptions.find_one({'_id': ObjectId(jd_id)})
        if jd_data:
            return JobDescription(jd_data)
        return None
    
    @staticmethod
    def get_by_recruiter_id(recruiter_id):
        jds = mongo.db.job_descriptions.find({'recruiter_id': recruiter_id})
        return [JobDescription(jd) for jd in jds]
