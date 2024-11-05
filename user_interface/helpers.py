from flask import redirect, render_template, session
from functools import wraps


def apology_login(message, code=400):
    """Render message as an apology to user."""

    return render_template("apology_login.html", code=code, message=message), code


def apology_home(message, code=400):
    """Render message as an apology to user."""

    return render_template("apology_home.html", code=code, message=message), code


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


