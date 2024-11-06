# app.py
import os
import sys
# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import json
import sqlite3
from flask import Flask, flash, redirect, render_template, request, session, send_from_directory, g, url_for, current_app
from flask_session import Session
from werkzeug.security import check_password_hash, generate_password_hash

from helpers import apology_login, login_required, get_db
import main
from main import get_resumes, resume_extraction, job_info_extraction, calc_similarity

# Import the blueprint from the recruitment package
from recruitment import recruitment_bp

# Configure application
app = Flask(__name__)

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# File upload configuration
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Register the blueprint
app.register_blueprint(recruitment_bp)

# Database teardown function
@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()

@app.after_request
def after_request(response):
    """Ensure responses aren't cached."""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

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
            api_key = os.getenv("OPENAI_API_KEY", "<your OpenAI API key if not set as an env var>")

            # Retrieve bespoke apologies
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
    """Display tailored interviews for selected applicants."""
    # Fetch data
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM applicants WHERE interview_status = 'Selected'")
    winners = cursor.fetchall()
    cursor.execute("SELECT required_skills FROM job_postings")
    required_skills = cursor.fetchall()

    if request.method == 'POST':
        if winners:
            MODEL = "gpt-4o-mini"
            api_key = os.getenv("OPENAI_API_KEY", "<your OpenAI API key if not set as an env var>")

            # Retrieve tailored interview questions
            response = main.tailored_questions(api_key, winners, required_skills, model=MODEL)
        else:
            response = []
        winners = []
        required_skills = []
        return render_template("tailored_interviews.html", winners=winners, response=response, required_skills=required_skills)
    else:
        return render_template("tailored_interviews.html", winners=winners, required_skills=required_skills)

@app.route("/uploads/<name>")
@login_required
def download_file(name):
    """Send file from the uploads folder."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], name)

if __name__ == "__main__":
    app.run(debug=True)
