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
from ast import literal_eval

roles_bp = Blueprint('roles', __name__, template_folder='../templates')

@roles_bp.route('/roles', methods=['GET'])
@login_required
def roles():
    user_id = session.get('user_id')
    if not user_id:
        flash('User not identified')
        return redirect(url_for('auth.login'))  # Redirect to login if user_id is missing

    upload_folder = current_app.config['UPLOAD_FOLDER_ROLES']
    filename = f"user_{user_id}_resume.pdf"
    upload_path = os.path.join(upload_folder, filename)

    if os.path.exists(upload_path):
        applicant = filename  # This will be used to display or link to the resume
    else:
        applicant = None  # No resume uploaded yet

    db = get_db()
    cursor = db.cursor()
    cursor.execute("""
        SELECT role
        FROM unique_roles
        ORDER BY role ASC
    """)
    roles = [role[0] for role in cursor.fetchall()]

    return render_template(
        "roles.html",
        applicant=applicant,
        roles=roles
    )


@roles_bp.route('/roles/upload_files', methods=['POST'])
@login_required
def upload_files():
    file = request.files.get('file')
    if not file or file.filename == '':
        flash('No file selected')
        return redirect(url_for('roles.roles'))

    if allowed_file(file.filename):
        filename = secure_filename(file.filename)
        user_id = session.get('user_id')
        # Ensure user_id is available
        if not user_id:
            flash('User not identified')
            return redirect(url_for('roles.roles'))

        # Save the file with a unique name per user
        filename = f"user_{user_id}_resume.pdf"
        upload_path = os.path.join(current_app.config['UPLOAD_FOLDER_ROLES'], filename)

        # Optionally, remove the old resume if it exists
        if os.path.exists(upload_path):
            os.remove(upload_path)

        file.save(upload_path)
        flash('Resume successfully uploaded')
    else:
        flash(f'File type not allowed: {file.filename}')
    return redirect(url_for('roles.roles'))



@roles_bp.route('/roles/run_analysis', methods=['POST'])
@login_required
def run_analysis():
    # Get pre-loaded DataFrames
    growth_df = current_app.config['GROWTH_DF']
    df_skill_role_grouped = current_app.config['DF_SKILL_ROLE_GROUPED']
    similarity_df = current_app.config['SIMILARITY_DF']
    titles_df = current_app.config['TITLES_DF']
    merged_df = current_app.config['MERGED_DF']

    # Now you can proceed directly with analysis
    role = request.form.get('current_role')
    wage = float(request.form.get('current_salary'))
    
    df_resumes = main.get_resumes(current_app.config['UPLOAD_FOLDER_ROLES'])
    df_resumes = main.resume_extraction(df_resumes)
    foo, bar = main.role_similarity(df_resumes, merged_df, df_skill_role_grouped, similarity_df, titles_df, growth_df, role=role, wage=wage)

    foo_col = foo.columns
    bar_col = bar.columns
    foo = foo.head(5).to_dict(orient='records')
    bar = bar.head(5).to_dict(orient='records')

    return render_template(
        "roles.html",
        applicant=f"user_{session.get('user_id')}_resume.pdf",
        foo=foo,
        foo_col=foo_col,
        bar=bar,
        bar_col=bar_col,
        analysis_done=True
    )

@roles_bp.route('/roles/clear', methods=['POST'])
@login_required
def clear_results():
    # Get pre-loaded DataFrames
    upload_folder = current_app.config['UPLOAD_FOLDER_ROLES']
    try:
        # Delete all files in the uploads directory
        for filename in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        flash('All uploaded files and results have been cleared.')
    except Exception as e:
        flash(f'An error occurred while clearing files: {e}')
    


    return redirect(url_for('roles.roles'))