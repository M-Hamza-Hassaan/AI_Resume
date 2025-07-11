from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_required, current_user
from app.models.job_description import JobDescription
from app.models.resume import Resume
from app.nlp.parser import parse_job_description
from app.nlp.vectorizer import ResumeVectorizer
from app.nlp.ranker import ResumeRanker
from app.forms.job_description_form import JobDescriptionForm

recruiter = Blueprint('recruiter', __name__)

@recruiter.route('/recruiter/dashboard')
@login_required
def dashboard():
    if current_user.role != 'recruiter':
        flash('Access denied. You are not a recruiter.', 'danger')
        return redirect(url_for('main.index'))
    
    # Get all job descriptions for this recruiter
    job_descriptions = JobDescription.get_by_recruiter_id(current_user.id)
    
    return render_template('recruiter/dashboard.html', title='Recruiter Dashboard', job_descriptions=job_descriptions)

@recruiter.route('/recruiter/add-job-description', methods=['GET', 'POST'])
@login_required
def add_job_description():
    if current_user.role != 'recruiter':
        flash('Access denied. You are not a recruiter.', 'danger')
        return redirect(url_for('main.index'))
    
    form = JobDescriptionForm()
    
    if form.validate_on_submit():
        # Parse the job description
        parsed_data = parse_job_description(form.description.data)
        
        # Create job description
        jd = JobDescription()
        jd.recruiter_id = current_user.id
        jd.title = form.title.data
        jd.company = form.company.data
        jd.description = form.description.data
        jd.parsed_data = parsed_data
        
        # Vectorize the job description
        vectorizer = ResumeVectorizer()
        jd.vector = vectorizer.vectorize_job_description(parsed_data)
        
        # Save to database
        jd.save()
        
        flash('Job description added successfully!', 'success')
        return redirect(url_for('recruiter.dashboard'))
    
    return render_template('recruiter/add_job_description.html', title='Add Job Description', form=form)

@recruiter.route('/recruiter/view-job-description/<jd_id>')
@login_required
def view_job_description(jd_id):
    if current_user.role != 'recruiter':
        flash('Access denied. You are not a recruiter.', 'danger')
        return redirect(url_for('main.index'))
    
    # Get the job description
    jd = JobDescription.get_by_id(jd_id)
    
    if not jd or jd.recruiter_id != current_user.id:
        flash('Job description not found.', 'danger')
        return redirect(url_for('recruiter.dashboard'))
    
    return render_template('recruiter/view_job_description.html', title='View Job Description', jd=jd)

@recruiter.route('/recruiter/rank-resumes/<jd_id>')
@login_required
def rank_resumes(jd_id):
    if current_user.role != 'recruiter':
        flash('Access denied. You are not a recruiter.', 'danger')
        return redirect(url_for('main.index'))
    
    # Get the job description
    jd = JobDescription.get_by_id(jd_id)
    
    if not jd or jd.recruiter_id != current_user.id:
        flash('Job description not found.', 'danger')
        return redirect(url_for('recruiter.dashboard'))
    
    # Rank resumes
    ranker = ResumeRanker()
    ranked_resumes = ranker.rank_resumes_for_job(jd_id)
    
    return render_template('recruiter/ranked_resumes.html', title='Ranked Resumes', jd=jd, ranked_resumes=ranked_resumes)

@recruiter.route('/recruiter/resume-insights/<jd_id>/<resume_id>')
@login_required
def resume_insights(jd_id, resume_id):
    if current_user.role != 'recruiter':
        flash('Access denied. You are not a recruiter.', 'danger')
        return redirect(url_for('main.index'))
    
    # Get the job description
    jd = JobDescription.get_by_id(jd_id)
    
    if not jd or jd.recruiter_id != current_user.id:
        flash('Job description not found.', 'danger')
        return redirect(url_for('recruiter.dashboard'))
    
    # Get the resume
    resume = Resume.get_by_id(resume_id)
    
    if not resume:
        flash('Resume not found.', 'danger')
        return redirect(url_for('recruiter.rank_resumes', jd_id=jd_id))
    
    # Get insights
    ranker = ResumeRanker()
    insights = ranker.get_resume_insights(resume_id, jd_id)
    
    return render_template('recruiter/resume_insights.html', title='Resume Insights', jd=jd, resume=resume, insights=insights)
