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
import datetime

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
        db.execute('PRAGMA foreign_keys = ON;')
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# File upload configuration
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
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
            applicant_files = [file for file in os.listdir(upload_folder) if file.endswith('.pdf')]

            if not applicant_files:
                flash('No applicants to analyze')
                return redirect(url_for('ext_recruit'))

            if not job_description:
                flash('No job description provided')
                return redirect(url_for('ext_recruit'))

            db = get_db()
            cursor = db.cursor()

            # Step 1: Insert Job Posting into the Database
            # Extract skills from the job description
            df_jobs = pd.DataFrame([{'raw': job_description}])
            df_jobs = main.job_info_extraction(df_jobs)

            required_skills = df_jobs.loc[0, 'Skills']
            required_skills_json = json.dumps(required_skills)

            # Insert job posting into the database
            cursor.execute("""
                INSERT INTO job_postings (job_description, required_skills)
                VALUES (?, ?)
            """, (job_description, required_skills_json))
            job_id = cursor.lastrowid  # Get the job_id of the inserted job

            # Step 2: Insert Applicants into the Database
            # Extract applicant data
            df_resumes = main.get_resumes(upload_folder)
            df_resumes = main.resume_extraction(df_resumes)

            for _, row in df_resumes.iterrows():
                name = row.get('name', '')
                contact_info = ''  # Update as needed
                skills = row.get('Skills', [])
                skills_json = json.dumps(skills)
                degrees = row.get('Degrees', [])
                degrees_json = json.dumps(degrees)

                # Insert or update applicant in the database
                cursor.execute("""
                    INSERT OR REPLACE INTO applicants (name, contact_info, skills, degrees)
                    VALUES (?, ?, ?, ?)
                """, (name, contact_info, skills_json, degrees_json))

            # Commit the inserts
            db.commit()

            # Step 3: Perform Similarity Analysis
            # Calculate similarity
            matching_dataframe = main.calc_similarity(df_resumes, df_jobs)

            # Update applicants with similarity scores and ranks
            for _, row in matching_dataframe.iterrows():
                name = row['applicant']
                similarity_score = row['all-mpnet-base-v2_score']
                rank = int(row['rank'])
                interview_status = row['interview_status']

                cursor.execute("""
                    UPDATE applicants
                    SET similarity_score = ?, rank = ?, interview_status = ?
                    WHERE name = ?
                """, (similarity_score, rank, interview_status, name))

            # Commit the updates
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

            return render_template("ext_recruit.html", applicants=applicant_files, analysis_data=analysis_data, columns=columns)

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

        flash('All uploaded files and results have been cleared.')
    except Exception as e:
        flash(f'An error occurred while clearing files: {e}')
    return redirect(url_for('ext_recruit'))


@app.route("/bespoke_apology", methods=["GET", "POST"])
@login_required
def bespoke_apology():
    """ Endpoint for displaying the bespoke apologies for the applicants that didn't make it to the interviews """
    
    # Fetch Data
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM applicants WHERE interview_status = 'Not Selected'")
    losers = cursor.fetchall()
    cursor.execute("SELECT required_skills FROM job_postings")
    required_skills = cursor.fetchall()
    

    if request.method == 'POST':
        if losers:
            MODEL="gpt-4o-mini"
            api_key=os.getenv("OPENAI_API_KEY", "<your OpenAI API key if not set as an env var>")

            # Retrieve tailored questions
            response = main.bespoke_apologies(api_key, losers, required_skills, model=MODEL)
        else:
            response = []
        losers = []
        required_skills = []
        return render_template("bespoke_apology.html", losers=losers, response=response, required_skills=required_skills)
    else:
        return render_template("bespoke_apology.html", losers=losers, required_skills=required_skills)

@app.route("/tailored_interviews", methods=["GET", "POST"])
@login_required
def tailored_interviews():
    """ Endpoint for displaying the tailored interviews for the applicants that made it to the interviews """
    
    # Fetch data
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM applicants WHERE interview_status = 'Selected'")
    winners = cursor.fetchall()
    cursor.execute("SELECT required_skills FROM job_postings")
    required_skills = cursor.fetchall()


    if request.method == 'POST':
        if winners:
            MODEL="gpt-4o-mini"
            api_key=os.getenv("OPENAI_API_KEY", "<your OpenAI API key if not set as an env var>")

            # Retrieve tailored questions
            response = main.tailored_questions(api_key, winners, required_skills, model=MODEL)
        else:
            response = []
        winners = []
        required_skills = []
        return render_template("tailored_interviews.html", winners=winners, response=response, required_skills=required_skills)
    else:
        return render_template("tailored_interviews.html", winners=winners, required_skills=required_skills )

    

@app.route("/uploads/<name>")
@login_required
def download_file(name):
    """Send file from the uploads folder"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], name)

if __name__ == "__main__":
    app.run(debug=True)