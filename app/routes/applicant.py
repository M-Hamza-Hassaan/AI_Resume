from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
import uuid
from app.models.resume import Resume
from app.nlp.parser import parse_resume
from app.forms.resume_form import ResumeUploadForm

applicant = Blueprint('applicant', __name__)


@applicant.route('/applicant/dashboard')
@login_required
def dashboard():
    if current_user.role != 'applicant':
        flash('Access denied. You are not an applicant.', 'danger')
        return redirect(url_for('main.index'))

    resume = None
    if current_user.resume_id:
        resume = Resume.get_by_id(current_user.resume_id)

    return render_template('applicant/dashboard.html',
                           title='Applicant Dashboard', resume=resume)


@applicant.route('/applicant/upload-resume', methods=['GET', 'POST'])
@login_required
def upload_resume():
    if current_user.role != 'applicant':
        flash('Access denied. You are not an applicant.', 'danger')
        return redirect(url_for('main.index'))

    form = ResumeUploadForm()

    if form.validate_on_submit():
        resume_file = form.resume.data
        original_filename = secure_filename(resume_file.filename)

        # Prevent filename collisions — prefix with a UUID
        ext = os.path.splitext(original_filename)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}{ext}"
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
        resume_file.save(file_path)

        try:
            # Parse the resume with custom NER + regex
            parsed_data = parse_resume(file_path)

            # Retrieve or create resume record for this user
            resume = Resume.get_by_user_id(current_user.id) or Resume()
            resume.user_id = current_user.id
            resume.filename = original_filename   # store original name for display
            resume.parsed_data = parsed_data

            # Vectors are rebuilt over the current candidate pool at ranking time.
            resume.vector = None

            resume.save()

            skills_found = len(parsed_data.get('skills', []))
            flash(
                f'Resume uploaded and parsed successfully! '
                f'Detected {skills_found} skills.',
                'success'
            )
            return redirect(url_for('applicant.dashboard'))

        except Exception as e:
            # Remove the saved file if processing failed
            if os.path.exists(file_path):
                os.remove(file_path)
            flash(f'Error processing resume: {str(e)}', 'danger')

    return render_template('applicant/upload_resume.html',
                           title='Upload Resume', form=form)


@applicant.route('/applicant/view-resume')
@login_required
def view_resume():
    if current_user.role != 'applicant':
        flash('Access denied. You are not an applicant.', 'danger')
        return redirect(url_for('main.index'))

    resume = None
    if current_user.resume_id:
        resume = Resume.get_by_id(current_user.resume_id)

    if not resume:
        flash('No resume found. Please upload your resume.', 'info')
        return redirect(url_for('applicant.upload_resume'))

    return render_template('applicant/view_resume.html',
                           title='View Resume', resume=resume)
