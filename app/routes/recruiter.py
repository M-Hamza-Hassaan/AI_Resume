import csv
import io

from flask import (
    Blueprint,
    Response,
    flash,
    redirect,
    render_template,
    url_for,
)
from flask_login import login_required, current_user
from app.models.job_description import JobDescription
from app.models.resume import Resume
from app.nlp.parser import parse_job_description
from app.nlp.ranker import ResumeRanker, SBERT_AVAILABLE
from app.nlp.fairness import FairnessAuditor
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

        # TF-IDF vectors are rebuilt with a shared corpus at ranking time.
        jd.vector = None

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

    # Rank resumes — enable SBERT if available, enable anonymization for fairness
    ranker = ResumeRanker(
        use_sbert=SBERT_AVAILABLE,
        toggle_anonymization=True,
    )
    ranked_resumes = ranker.rank_resumes_for_job(jd_id)

    return render_template(
        'recruiter/ranked_resumes.html',
        title='Ranked Resumes', jd=jd, ranked_resumes=ranked_resumes,
        sbert_enabled=SBERT_AVAILABLE,
    )


@recruiter.route('/recruiter/export-rankings/<jd_id>')
@login_required
def export_rankings(jd_id):
    if current_user.role != 'recruiter':
        flash('Access denied. You are not a recruiter.', 'danger')
        return redirect(url_for('main.index'))

    jd = JobDescription.get_by_id(jd_id)
    if not jd or jd.recruiter_id != current_user.id:
        flash('Job description not found.', 'danger')
        return redirect(url_for('recruiter.dashboard'))

    ranker = ResumeRanker(
        use_sbert=SBERT_AVAILABLE,
        toggle_anonymization=True,
    )
    ranked_resumes = ranker.rank_resumes_for_job(jd_id)
    tfidf_ranks = {
        resume['resume_id']: rank
        for rank, resume in enumerate(
            sorted(ranked_resumes, key=lambda item: item['tfidf_score'], reverse=True),
            start=1,
        )
    }
    sbert_ranked = [
        resume for resume in ranked_resumes
        if resume['sbert_score'] is not None
    ]
    sbert_ranks = {
        resume['resume_id']: rank
        for rank, resume in enumerate(
            sorted(sbert_ranked, key=lambda item: item['sbert_score'], reverse=True),
            start=1,
        )
    }

    output = io.StringIO(newline='')
    fields = (
        'combined_rank',
        'tfidf_rank',
        'sbert_rank',
        'resume_id',
        'combined_score',
        'tfidf_score',
        'sbert_score',
        'skills_score',
        'experience_score',
        'education_score',
    )
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for rank, resume in enumerate(ranked_resumes, start=1):
        writer.writerow({
            'combined_rank': rank,
            'tfidf_rank': tfidf_ranks[resume['resume_id']],
            'sbert_rank': sbert_ranks.get(resume['resume_id'], ''),
            'resume_id': resume['resume_id'],
            'combined_score': resume['combined_score'],
            'tfidf_score': resume['tfidf_score'],
            'sbert_score': (
                resume['sbert_score']
                if resume['sbert_score'] is not None else ''
            ),
            'skills_score': resume['skills_score'],
            'experience_score': resume['experience_score'],
            'education_score': resume['education_score'],
        })

    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={
            'Content-Disposition':
                f'attachment; filename="resume-rankings-{jd_id}.csv"'
        },
    )


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

    # Get insights with dual-model scoring + XAI + fairness
    ranker = ResumeRanker(
        use_sbert=SBERT_AVAILABLE,
        toggle_anonymization=True,
    )
    insights = ranker.get_resume_insights(resume_id, jd_id)

    # Run fairness / counterfactual audit on this resume
    fairness_result = None
    if insights and resume.parsed_data.get('full_text'):
        auditor = FairnessAuditor(ranker=ranker)
        fairness_result = auditor.counterfactual_test(
            resume.parsed_data.get('full_text', ''),
            jd.parsed_data.get('full_text', ''),
            matching_fn=lambda resume_text, jd_text: ranker.compute_score_for_text(
                resume_text,
                jd_text,
                insights['skills_score'],
                insights['experience_score'],
                insights['education_score'],
            ),
        )

    return render_template(
        'recruiter/resume_insights.html',
        title='Resume Insights', jd=jd, resume=resume, insights=insights,
        fairness_result=fairness_result,
        sbert_enabled=SBERT_AVAILABLE,
    )
