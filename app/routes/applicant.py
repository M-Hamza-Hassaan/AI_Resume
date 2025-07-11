from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
from app.models.resume import Resume
from app.nlp.parser import parse_resume
from app.nlp.vectorizer import ResumeVectorizer
from app.forms.resume_form import ResumeUploadForm

applicant = Blueprint('applicant', __name__)

@applicant.route('/applicant/dashboard')
@login_required
def dashboard():
    if current_user.role != 'applicant':
        flash('Access denied. You are not an applicant.', 'danger')
        return redirect(url_for('main.index'))
    
    # Get the applicant's resume
    resume = None
    if current_user.resume_id:
        resume = Resume.get_by_id(current_user.resume_id)
    
    return render_template('applicant/dashboard.html', title='Applicant Dashboard', resume=resume)

@applicant.route('/applicant/upload-resume', methods=['GET', 'POST'])
@login_required
def upload_resume():
    if current_user.role != 'applicant':
        flash('Access denied. You are not an applicant.', 'danger')
        return redirect(url_for('main.index'))
    
    form = ResumeUploadForm()
    
    if form.validate_on_submit():
        # Save the uploaded file
        resume_file = form.resume.data
        filename = secure_filename(resume_file.filename)
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        resume_file.save(file_path)
        
        try:
            # Parse the resume
            parsed_data = parse_resume(file_path)
            
            # Create or update resume
            resume = Resume.get_by_user_id(current_user.id) or Resume()
            resume.user_id = current_user.id
            resume.filename = filename
            resume.parsed_data = parsed_data
            
            # Vectorize the resume
            vectorizer = ResumeVectorizer()
            resume.vector = vectorizer.vectorize_resume(parsed_data)
            
            # Save to database
            resume.save()
            
            flash('Resume uploaded and parsed successfully!', 'success')
            return redirect(url_for('applicant.dashboard'))
            
        except Exception as e:
            flash(f'Error processing resume: {str(e)}', 'danger')
    
    return render_template('applicant/upload_resume.html', title='Upload Resume', form=form)

@applicant.route('/applicant/view-resume')
@login_required
def view_resume():
    if current_user.role != 'applicant':
        flash('Access denied. You are not an applicant.', 'danger')
        return redirect(url_for('main.index'))
    
    # Get the applicant's resume
    resume = None
    if current_user.resume_id:
        resume = Resume.get_by_id(current_user.resume_id)
    
    if not resume:
        flash('No resume found. Please upload your resume.', 'info')
        return redirect(url_for('applicant.upload_resume'))
    
    return render_template('applicant/view_resume.html', title='View Resume', resume=resume)
