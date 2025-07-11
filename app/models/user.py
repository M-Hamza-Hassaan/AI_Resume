from flask_login import UserMixin
from app import mongo, login_manager
from bson.objectid import ObjectId

@login_manager.user_loader
def load_user(user_id):
    user_data = mongo.db.users.find_one({"_id": ObjectId(user_id)})
    if user_data:
        return User(user_data)
    return None

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data.get('_id'))
        self.email = user_data.get('email')
        self.name = user_data.get('name')
        self.role = user_data.get('role')  # 'applicant' or 'recruiter'
        self.password = user_data.get('password')
        self.resume_id = user_data.get('resume_id')  # For applicants
        self.job_descriptions = user_data.get('job_descriptions', [])  # For recruiters
    
    def get_id(self):
        return self.id
    
    @staticmethod
    def get_by_email(email):
        user_data = mongo.db.users.find_one({"email": email})
        if user_data:
            return User(user_data)
        return None
