from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SubmitField
from wtforms.validators import DataRequired, Length

class JobDescriptionForm(FlaskForm):
    title = StringField('Job Title', validators=[DataRequired(), Length(min=2, max=100)])
    company = StringField('Company', validators=[DataRequired(), Length(min=2, max=100)])
    description = TextAreaField('Job Description', validators=[DataRequired(), Length(min=50)])
    submit = SubmitField('Add Job Description')
