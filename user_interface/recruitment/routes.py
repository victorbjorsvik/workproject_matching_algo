import os
import sys
# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Blueprint, render_template, request, redirect, url_for, flash, session, current_app
from helpers import allowed_file, get_db, login_required
from werkzeug.utils import secure_filename
import json
import pandas as pd
import main 

recruitment_bp = Blueprint('recruitment', __name__, template_folder='../templates')

@recruitment_bp.route('/ext_recruit', methods=['GET'])
@login_required
def ext_recruit():
    upload_folder = current_app.config['UPLOAD_FOLDER_EXT']
    applicants = [file for file in os.listdir(upload_folder) if file.endswith('.pdf')]
    job_description = session.get('job_description', '')

    # Initialize analysis_data as empty
    analysis_data = []
    columns = []

    # Connect to the database
    db = get_db()
    cursor = db.cursor()

    # Check if analysis results exist in the database
    cursor.execute("""
        SELECT name, similarity_score, rank, interview_status
        FROM applicants
        WHERE similarity_score IS NOT NULL
        ORDER BY rank ASC
    """)
    ranked_applicants = cursor.fetchall()

    if ranked_applicants:
        # Prepare data for template
        analysis_data = [{
            'name': row[0],
            'similarity_score': round(row[1], 3),
            'rank': row[2],
            'interview_status': row[3]
        } for row in ranked_applicants]
        columns = ['name', 'similarity_score', 'rank', 'interview_status']

    return render_template(
        "ext_recruit.html",
        applicants=applicants,
        analysis_data=analysis_data,
        job_description=job_description,
        columns=columns
    )

@recruitment_bp.route('/ext_recruit/upload_files', methods=['POST'])
@login_required
def upload_files():
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        flash('No files selected')
        return redirect(url_for('recruitment.ext_recruit'))

    for file in files:
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(current_app.config['UPLOAD_FOLDER_EXT'], filename)
            file.save(upload_path)
        else:
            flash(f'File type not allowed: {file.filename}')
            return redirect(url_for('recruitment.ext_recruit'))

    flash('Files successfully uploaded')
    return redirect(url_for('recruitment.ext_recruit'))

@recruitment_bp.route('/ext_recruit/submit_job_description', methods=['POST'])
@login_required
def submit_job_description():
    job_description = request.form.get('job_description', '')
    if not job_description:
        flash('No job description provided')
        return redirect(url_for('recruitment.ext_recruit'))

    session['job_description'] = job_description
    flash('Job description saved')
    return redirect(url_for('recruitment.ext_recruit'))

@recruitment_bp.route('/ext_recruit/run_analysis', methods=['POST'])
@login_required
def run_analysis():
    job_description = session.get('job_description')
    upload_folder = current_app.config['UPLOAD_FOLDER_EXT']
    applicant_files = [file for file in os.listdir(upload_folder) if file.endswith('.pdf')]

    if not applicant_files:
        flash('No applicants to analyze')
        return redirect(url_for('recruitment.ext_recruit'))

    if not job_description:
        flash('No job description provided')
        return redirect(url_for('recruitment.ext_recruit'))

    # Retrieve the number of applicants from the form data
    num_applicants_str = request.form.get('num_applicants', '3')
    try:
        num_applicants = int(num_applicants_str)
        if num_applicants < 1:
            num_applicants = 1
    except ValueError:
        flash('Invalid number of applicants. Using default value of 3.')
        num_applicants = 3

    db = get_db()
    cursor = db.cursor()

    # Rest of your code remains the same, but pass num_applicants to calc_similarity
    # Step 1: Insert Job Posting into the Database
    df_jobs = pd.DataFrame([{'raw': job_description}])
    df_jobs = main.job_info_extraction(df_jobs)

    required_skills = df_jobs.loc[0, 'Skills']
    required_skills_json = json.dumps(required_skills)

    cursor.execute("""
        INSERT INTO job_postings (job_description, required_skills)
        VALUES (?, ?)
    """, (job_description, required_skills_json))
    job_id = cursor.lastrowid

    # Step 2: Insert Applicants into the Database
    df_resumes = main.get_resumes(upload_folder)
    df_resumes = main.resume_extraction(df_resumes)

    for _, row in df_resumes.iterrows():
        name = row.get('name', '')
        contact_info = ''  # Update as needed
        skills = row.get('Skills', [])
        skills_json = json.dumps(skills)
        degrees = row.get('Degrees', [])
        degrees_json = json.dumps(degrees)

        cursor.execute("""
            INSERT OR REPLACE INTO applicants (name, contact_info, skills, degrees)
            VALUES (?, ?, ?, ?)
        """, (name, contact_info, skills_json, degrees_json))

    db.commit()

    # Step 3: Perform Similarity Analysis
    matching_dataframe = main.calc_similarity(df_resumes, df_jobs, N=num_applicants, parallel=True)

    # Update applicants with similarity scores and ranks
    for _, row in matching_dataframe.iterrows():
        name = row['name']
        similarity_score = row['similarity_score']
        rank = int(row['rank'])
        interview_status = row['interview_status']

        cursor.execute("""
            UPDATE applicants
            SET similarity_score = ?, rank = ?, interview_status = ?
            WHERE name = ?
        """, (similarity_score, rank, interview_status, name))

    db.commit()

    # Step 4: Retrieve Ranked Applicants for Display
    cursor.execute("""
        SELECT name, similarity_score, rank, interview_status
        FROM applicants
        ORDER BY rank ASC
    """)
    ranked_applicants = cursor.fetchall()

    # Prepare data for template
    analysis_data = [{
        'name': row[0],
        'similarity_score': round(row[1], 3),
        'rank': row[2],
        'interview_status': row[3]
    } for row in ranked_applicants]

    columns = ['name', 'similarity_score', 'rank', 'interview_status']

    return render_template(
        "ext_recruit.html",
        applicants=applicant_files,
        analysis_data=analysis_data,
        job_description=job_description,
        columns=columns,
        num_applicants=num_applicants 
    )


@recruitment_bp.route('/ext_recruit/clear', methods=['POST'])
@login_required
def clear_results():
    upload_folder = current_app.config['UPLOAD_FOLDER_EXT']
    db = get_db()
    cursor = db.cursor()
    try:
        # Delete all files in the uploads directory
        for filename in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)

        # Clear database tables
        cursor.execute("DELETE FROM applicants")
        cursor.execute("DELETE FROM job_postings")
        db.commit()

        # Clear session data
        session.pop('job_description', None)

        flash('All uploaded files and results have been cleared.')
    except Exception as e:
        flash(f'An error occurred while clearing files: {e}')
    return redirect(url_for('recruitment.ext_recruit'))
