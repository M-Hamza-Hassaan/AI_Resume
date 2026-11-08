from flask import Flask
from flask_pymongo import PyMongo
from flask_login import LoginManager
from flask_bcrypt import Bcrypt
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

mongo = PyMongo()
login_manager = LoginManager()
bcrypt = Bcrypt()


def create_app(config_class=None):
    app = Flask(__name__)

    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev_key_change_in_production")
    app.config["MONGO_URI"] = os.environ.get("MONGO_URI", "mongodb://localhost:27017/airesume")
    app.config["UPLOAD_FOLDER"] = os.path.join(app.root_path, "uploads")
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Initialize extensions
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

    # ------------------------------------------------------------------ #
    #  Post-init: ensure mongo.db is set + create indexes                 #
    # ------------------------------------------------------------------ #
    with app.app_context():
        # Flask-PyMongo 3.x sets mongo.db=None when the database name
        # is not parsed from the URI (common with Atlas SRV strings).
        # Explicitly set it here using the MongoClient.
        if mongo.db is None and mongo.cx is not None:
            db_name = _parse_db_name(app.config["MONGO_URI"])
            mongo.db = mongo.cx[db_name]
            logger.info("mongo.db explicitly set to database: %s", db_name)

        if mongo.db is not None:
            _create_indexes()
        else:
            logger.warning(
                "mongo.db is None — MongoDB connection may not be established. "
                "Check MONGO_URI and Atlas network access settings."
            )

    return app


def _parse_db_name(uri: str) -> str:
    """
    Extract the database name from a MongoDB URI.

    Falls back to 'airesume' if the name cannot be parsed.
    Handles both standard and SRV URIs.
    """
    try:
        from pymongo.uri_parser import parse_uri
        parsed = parse_uri(uri)
        db = parsed.get("database")
        if db:
            return db
    except Exception:
        pass

    # Regex fallback: extract the path component between "/" and "?"
    import re
    match = re.search(r"\.(?:mongodb\.net|localhost|[\w.]+)/([^/?]+)", uri)
    if match:
        db = match.group(1).strip()
        if db:
            return db

    return "airesume"


def _create_indexes():
    """
    Create MongoDB indexes for performance and data integrity.

    - Unique index on users.email — prevents duplicate registrations.
    - Index on resumes.user_id — fast lookup by user.
    - Index on job_descriptions.recruiter_id — fast lookup by recruiter.
    """
    try:
        mongo.db.users.create_index("email", unique=True, sparse=True)
        mongo.db.resumes.create_index("user_id")
        mongo.db.job_descriptions.create_index("recruiter_id")
        logger.info("MongoDB indexes ensured.")
    except Exception as e:
        logger.warning("Could not create MongoDB indexes: %s", e)
