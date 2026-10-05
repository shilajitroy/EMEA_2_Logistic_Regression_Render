import hmac
import math
import os
import secrets
from functools import wraps
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, redirect, render_template, request, session, url_for

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)

DEMO_USERNAME = os.environ.get("DEMO_USERNAME", "viewer")
DEMO_PASSWORD = os.environ.get("DEMO_PASSWORD", "adclick123")

MODEL_PATH = Path(__file__).resolve().parent / "ad_click_model.joblib"
model = joblib.load(MODEL_PATH)


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped_view


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if hmac.compare_digest(username, DEMO_USERNAME) and hmac.compare_digest(
            password, DEMO_PASSWORD
        ):
            session.clear()
            session["logged_in"] = True
            session["username"] = DEMO_USERNAME
            return redirect(url_for("dashboard"))
        error = "The username or password was not correct."

    return render_template(
        "login.html",
        error=error,
        demo_username=DEMO_USERNAME,
        demo_password=DEMO_PASSWORD,
    )


@app.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def dashboard():
    return render_template(
        "index.html",
        username=session.get("username", DEMO_USERNAME),
        last_prediction=session.get("last_prediction"),
    )


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    error = None
    result = None
    values = {
        "site_time": "65",
        "age": "36",
        "income": "55000",
        "internet_usage": "180",
        "gender": "Female",
    }

    if request.method == "POST":
        values.update({key: request.form.get(key, "") for key in values})
        try:
            site_time = float(values["site_time"])
            age = int(values["age"])
            income = float(values["income"])
            internet_usage = float(values["internet_usage"])
            gender = values["gender"]

            if not all(math.isfinite(value) for value in (site_time, income, internet_usage)):
                raise ValueError("Inputs must be finite numbers.")
            if site_time < 0 or income < 0 or internet_usage < 0:
                raise ValueError("Inputs cannot be negative.")
            if not 18 <= age <= 100:
                raise ValueError("Age must be between 18 and 100.")
            if gender not in {"Female", "Male"}:
                raise ValueError("Select a valid gender.")

            input_data = pd.DataFrame([{
                "Daily Time Spent on Site": site_time,
                "Age": age,
                "Area Income": income,
                "Daily Internet Usage": internet_usage,
                "Male": 1 if gender == "Male" else 0,
            }])

            prediction = model.predict(input_data)[0]
            probability = float(model.predict_proba(input_data)[0, 1])
            result = {
                "clicked": bool(prediction == 1),
                "message": (
                    "The model predicts this user will click the ad."
                    if prediction == 1
                    else "The model predicts this user will not click the ad."
                ),
                "probability": f"{probability:.1%}",
            }
            session["last_prediction"] = result
        except (KeyError, TypeError, ValueError, OverflowError):
            error = "Enter valid values: non-negative numbers and an age from 18 to 100."

    return render_template("predict.html", result=result, error=error, values=values)


@app.route("/about")
@login_required
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run()
