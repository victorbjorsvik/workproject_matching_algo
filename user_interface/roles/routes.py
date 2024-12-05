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

    return render_template(
        "roles.html",
        applicant=applicant
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
