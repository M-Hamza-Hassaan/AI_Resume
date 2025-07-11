from app import mongo
from bson.objectid import ObjectId
import datetime

class Resume:
    def __init__(self, resume_data=None):
        if resume_data:
            self.id = str(resume_data.get('_id'))
            self.user_id = resume_data.get('user_id')
            self.filename = resume_data.get('filename')
            self.upload_date = resume_data.get('upload_date')
            self.parsed_data = resume_data.get('parsed_data', {})
            self.vector = resume_data.get('vector')
        else:
            self.id = None
            self.user_id = None
            self.filename = None
            self.upload_date = datetime.datetime.now()
            self.parsed_data = {}
            self.vector = None
    
    def save(self):
        resume_data = {
            'user_id': self.user_id,
            'filename': self.filename,
            'upload_date': self.upload_date,
            'parsed_data': self.parsed_data,
            'vector': self.vector
        }
        
        if self.id:
            mongo.db.resumes.update_one(
                {'_id': ObjectId(self.id)},
                {'$set': resume_data}
            )
        else:
            result = mongo.db.resumes.insert_one(resume_data)
            self.id = str(result.inserted_id)
        
        # Update user's resume_id
        mongo.db.users.update_one(
            {'_id': ObjectId(self.user_id)},
            {'$set': {'resume_id': self.id}}
        )
        
        return self.id
    
    @staticmethod
    def get_by_id(resume_id):
        resume_data = mongo.db.resumes.find_one({'_id': ObjectId(resume_id)})
        if resume_data:
            return Resume(resume_data)
        return None
    
    @staticmethod
    def get_by_user_id(user_id):
        resume_data = mongo.db.resumes.find_one({'user_id': user_id})
        if resume_data:
            return Resume(resume_data)
        return None
    
    @staticmethod
    def get_all():
        resumes = mongo.db.resumes.find()
        return [Resume(resume) for resume in resumes]
