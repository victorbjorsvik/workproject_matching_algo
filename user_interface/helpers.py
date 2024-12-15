from flask import redirect, render_template, session, g
from functools import wraps
import sqlite3

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect('skills.db')
        g.db.execute('PRAGMA foreign_keys = ON;')
    return g.db

def apology_login(message, code=400):
    """Render message as an apology to user."""
    return render_template("apology_login.html", code=code, message=message), code

def apology_openai(message, code=400):
    """Render message as an apology to user."""
    return render_template("apology_openai.html", code=code, message=message), code

def login_required(f):
    """
    Decorate routes to require login.

    https://flask.palletsprojects.com/en/1.1.x/patterns/viewdecorators/
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get("user_id") is None:
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated_function

def usd(value):
    """Format value as USD."""
    return f"${value:,.2f}"
