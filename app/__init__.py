from flask import Flask
from flask_pymongo import PyMongo
from flask_login import LoginManager
from flask_bcrypt import Bcrypt
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize extensions
mongo = PyMongo()
login_manager = LoginManager()
bcrypt = Bcrypt()

def create_app(config_class=None):
    app = Flask(__name__)
    
    # Configure app
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev_key_change_in_production")
    app.config["MONGO_URI"] = os.environ.get("MONGO_URI", "mongodb://localhost:27017/smart_hire")
    app.config["UPLOAD_FOLDER"] = os.path.join(app.root_path, "uploads")
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload
    
    # Ensure upload directory exists
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    
    # Initialize extensions with app
    mongo.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"
    login_manager.login_message_category = "info"
    bcrypt.init_app(app)
    
    # Register blueprints
    from app.routes.auth import auth
    from app.routes.main import main
    from app.routes.applicant import applicant
    from app.routes.recruiter import recruiter
    
    app.register_blueprint(auth)
    app.register_blueprint(main)
    app.register_blueprint(applicant)
    app.register_blueprint(recruiter)
    
    return app
