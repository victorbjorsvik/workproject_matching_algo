import os
import sqlite3
from flask import Flask, flash, redirect, render_template, request, session, jsonify, send_from_directory, g
from flask_session import Session
from werkzeug.security import check_password_hash, generate_password_hash
from helpers import apology_login, login_required

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

if __name__ == "__main__":
    app.run(debug=True)