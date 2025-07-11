from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField

class ResumeUploadForm(FlaskForm):
    resume = FileField('Upload Resume', validators=[
        FileRequired(),
        FileAllowed(['pdf', 'docx'], 'Only PDF and DOCX files are allowed!')
    ])
    submit = SubmitField('Upload')
