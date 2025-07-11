from flask import Blueprint, render_template, redirect, url_for
from flask_login import current_user

main = Blueprint('main  render_template, redirect, url_for', __name__)
from flask_login import current_user

main = Blueprint('main', __name__)

@main.route('/')
def index():
    if current_user.is_authenticated:
        if current_user.role == 'applicant':
            return redirect(url_for('applicant.dashboard'))
        else:
            return redirect(url_for('recruiter.dashboard'))
    
    return render_template('main/index.html', title='SmartHire')

@main.route('/about')
def about():
    return render_template('main/about.html', title='About SmartHire')
