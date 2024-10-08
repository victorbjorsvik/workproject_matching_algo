import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sqlite3
from flask import Flask, flash, redirect, render_template, request, session, jsonify, send_from_directory, g, url_for
from flask_session import Session
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from helpers import apology_login, login_required
import main
import pandas as pd
import json

# Configure application
app = Flask(__name__)

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Database configuration
DATABASE = 'skills.db'

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# File upload configuration
UPLOAD_FOLDER = '/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

@app.route("/")
@login_required
def index():
    """Render Welcome Page"""
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""
    session.clear()

    if request.method == "POST":
        if not request.form.get("username"):
            return apology_login("must provide username", 403)
        elif not request.form.get("password"):
            return apology_login("must provide password", 403)

        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (request.form.get("username"),))
        user = cursor.fetchone()

        if user is None or not check_password_hash(user[2], request.form.get("password")):
            return apology_login("invalid username and/or password", 403)

        session["user_id"] = user[0]
        return redirect("/")
    else:
        return render_template("login.html")

@app.route("/logout")
def logout():
    """Log user out"""
    session.clear()
    return redirect("/")

@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""
    if request.method == "POST":
        if not request.form.get("username"):
            return apology_login("must provide username", 400)
        elif not request.form.get("password"):
            return apology_login("must provide password", 400)
        if request.form.get("confirmation") != request.form.get("password"):
            return apology_login("Passwords must match", 400)

        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (request.form.get("username"),))
        if cursor.fetchone() is not None:
            return apology_login("Username already taken", 400)

        username = request.form.get("username")
        password = request.form.get("password")
        special = ['$', '@', '#', '!', '?', '*', '^', '=', '.', ',']

        if len(password) < 8:
            return apology_login("password must be at least 8 characters", 403)
        if len(password) > 20:
            return apology_login("password must be less than 20 character", 403)
        if not any(char.isdigit() for char in password):
            return apology_login("password must contain at least one digit", 403)
        if not any(char.isupper() for char in password):
            return apology_login("password must contain at least one uppercase letter", 403)
        if not any(char.islower() for char in password):
            return apology_login("password must contain at least one lowercase letter", 403)
        if not any(char in special for char in password):
            return apology_login("password must contain at least one special symbol", 403)

        hash = generate_password_hash(password)
        cursor.execute("INSERT INTO users (username, hash) VALUES (?, ?)", (username, hash))
        db.commit()

        return redirect("/login")
    else:
        return render_template("register.html")
    

@app.route("/ext_recruit", methods=["GET", "POST"])
@login_required
def ext_recruit():
    if request.method == 'POST':
        # Determine which form was submitted
        if 'files' in request.files:
            # Handle file uploads
            files = request.files.getlist('files')
            if not files or files[0].filename == '':
                flash('No files selected')
                return redirect(request.url)

            for file in files:
                if allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(upload_path)
                else:
                    flash(f'File type not allowed: {file.filename}')
                    return redirect(request.url)

            flash('Files successfully uploaded')
            return redirect(url_for('ext_recruit'))

        elif 'job_description' in request.form:
            # Handle job description submission
            job_description = request.form['job_description']
            session['job_description'] = job_description
            flash('Job description saved')
            return redirect(url_for('ext_recruit'))

        elif request.form.get('action') == 'run_analysis':
            # Handle running the analysis
            job_description = session.get('job_description')
            upload_folder = app.config['UPLOAD_FOLDER']
            applicants = [file for file in os.listdir(upload_folder) if file.endswith('.pdf')]

            if not applicants:
                flash('No applicants to analyze')
                return redirect(url_for('ext_recruit'))

            if not job_description:
                flash('No job description provided')
                return redirect(url_for('ext_recruit'))

            # Extract Skills, Degrees, and Majors from resumes
            resumes = main.get_resumes(upload_folder)
            res = main.resume_extraction(resumes)
            applicant_df = pd.read_json(res)

            # Extract skills, degrees, and majors from job description
            jobs = pd.DataFrame([job_description], columns=["raw"])
            jobs = main.job_info_extraction(jobs)
            job_df = pd.read_json(jobs)

            # Compare job description with applicants
            analysis_data_df = main.calc_similarity(applicant_df, job_df).sort_values(by='rank')
            analysis_data = analysis_data_df.to_dict(orient='records')

            return render_template("ext_recruit.html", applicants=applicants, analysis_data=analysis_data, analysis_data_df=analysis_data_df)

    else:
        # Handle GET request
        upload_folder = app.config['UPLOAD_FOLDER']
        applicants = [file for file in os.listdir(upload_folder) if file.endswith('.pdf')]
        job_description = session.get('job_description', '')

        # Initialize analysis_data as empty
        analysis_data = []

        return render_template("ext_recruit.html", applicants=applicants, analysis_data=analysis_data, job_description=job_description)



@app.route("/ext_recruit/clear", methods=["POST"])
@login_required
def clear_results():
    """Clear uploaded files and analysis results."""
    upload_folder = app.config['UPLOAD_FOLDER']
    try:
        # Delete all files in the uploads directory
        for filename in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        flash('All uploaded files and results have been cleared.')
    except Exception as e:
        flash(f'An error occurred while clearing files: {e}')
    return redirect(url_for('ext_recruit'))
    

@app.route("/uploads/<name>")
@login_required
def download_file(name):
    """Send file from the uploads folder"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], name)

if __name__ == "__main__":
    app.run(debug=True)