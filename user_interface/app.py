import os
import sys
from ast import literal_eval
# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import json
import sqlite3
from flask import Flask, redirect, render_template, request, session, send_from_directory, g, jsonify
from flask_session import Session
from werkzeug.security import check_password_hash, generate_password_hash
import tenacity

from helpers import apology_login, apology_openai, login_required, get_db, usd
import main

# Import blueprints for more extensie routes
from recruitment import recruitment_bp
from roles import roles_bp

# Configure application
app = Flask(__name__)

# Custom filter
app.jinja_env.filters["usd"] = usd

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# File upload configuration
UPLOAD_FOLDER_EXT = os.path.join(os.getcwd(), 'static', 'uploads', 'ext')
UPLOAD_FOLDER_ROLES = os.path.join(os.getcwd(),'static', 'uploads', 'roles')
os.makedirs(UPLOAD_FOLDER_EXT, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_ROLES, exist_ok=True)
app.config['UPLOAD_FOLDER_EXT'] = UPLOAD_FOLDER_EXT
app.config['UPLOAD_FOLDER_ROLES'] = UPLOAD_FOLDER_ROLES

##############################
# Pre-load and preprocess data
##############################
growth_df = pd.read_csv("data_from_girls/growth_df.csv")
df_skill_role_grouped = pd.read_csv("data_from_girls/hard_skills.csv")
similarity_df = pd.read_csv("data_from_girls/similarity_df.csv")
titles_df = pd.read_csv("data_from_girls/titles_df.csv")
merged_df = pd.read_csv("data_from_girls/merged_2.csv")

# Preprocess data once
merged_df["hourly_wage"] = merged_df["hourly_wage"].astype(float, errors="ignore")
merged_df["Skills"] = merged_df["Skills"].apply(literal_eval)
merged_df["skills_embed"] = merged_df["skills_embed"].apply(literal_eval)
df_skill_role_grouped["Skills"] = df_skill_role_grouped["Skills"].apply(literal_eval)

# Store data in app.config for easy access
app.config['GROWTH_DF'] = growth_df
app.config['DF_SKILL_ROLE_GROUPED'] = df_skill_role_grouped
app.config['SIMILARITY_DF'] = similarity_df
app.config['TITLES_DF'] = titles_df
app.config['MERGED_DF'] = merged_df
############################

# Register the blueprint
app.register_blueprint(recruitment_bp)
app.register_blueprint(roles_bp)

# Database teardown function
@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()


@app.route("/")
@login_required
def index():
    """Render Welcome Page."""
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in."""
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
    """Log user out."""
    session.clear()
    return redirect("/")

@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user."""
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
            return apology_login("password must be less than 20 characters", 403)
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

@app.route("/bespoke_apology", methods=["GET", "POST"])
@login_required
def bespoke_apology():
    """Display bespoke apologies for applicants not selected for interviews."""
    # Fetch data
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM applicants WHERE interview_status = 'Not Selected'")
    losers = cursor.fetchall()
    cursor.execute("SELECT required_skills FROM job_postings")
    required_skills = cursor.fetchall()

    if request.method == 'POST':
        if losers:
            MODEL = "gpt-4o-mini"

            if session.get("user_id") == 1:
                api_key = os.environ.get("OPENAI_API_KEY")
            else:
                api_key = request.form.get("password")
            try:
                response = main.bespoke_apologies(api_key, losers, required_skills, model=MODEL)
            except tenacity.RetryError as e:
                return apology_openai("Need a valid OpenAI API Key", 403)
    
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
    """Display tailored interviews for selected applicants."""
    # Fetch data
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        """SELECT * FROM applicants
          WHERE interview_status = 'Selected'
          ORDER BY RANK""")
    winners = cursor.fetchall()
    cursor.execute("SELECT required_skills FROM job_postings")
    required_skills = cursor.fetchall()

    if request.method == 'POST':
        if winners:
            MODEL = "gpt-4o-mini"

            if session.get("user_id") == 1:
                api_key = os.environ.get("OPENAI_API_KEY")
            else:
                api_key = request.form.get("password")
            try:
                # Retrieve tailored interview questions
                response = main.tailored_questions(api_key, winners, required_skills, model=MODEL)
            except tenacity.RetryError as e:
                return apology_openai("Need a valid OpenAI API Key", 403)
 
        else:
            response = []
        winners = []
        required_skills = []
        return render_template("tailored_interviews.html", winners=winners, response=response, required_skills=required_skills)
    else:
        return render_template("tailored_interviews.html", winners=winners, required_skills=required_skills)
    
@app.route("/tailored_interviews/coding_exercise", methods=["POST"])
@login_required
def coding_exercise():
    """Generate coding exercise based on job description."""
    # Fetch data
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT job_description FROM job_postings")
    job_description = cursor.fetchone()
    MODEL = "gpt-4o-mini"

    if session.get("user_id") == 1:
        api_key = os.environ.get("OPENAI_API_KEY")
    else:
        api_key = request.form.get("password")
    try:
        # Retrieve tailored interview questions
        coding_exercise_output= main.generate_coding_exercise(api_key, job_description, model=MODEL)
    except tenacity.RetryError as e:
        return jsonify({"error": "Invalid API Key"}), 403

    return jsonify({"coding_exercise": coding_exercise_output})


@app.route("/uploads/<name>")
@login_required
def download_file(name):
    """Send file from the uploads folder."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], name)

if __name__ == "__main__":
    app.run(debug=True)
