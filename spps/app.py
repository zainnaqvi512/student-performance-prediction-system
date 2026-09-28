"""Flask interface, persistent storage, authentication, and reports."""

from __future__ import annotations

import csv
import io
import json
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

import joblib
import pandas as pd
import psycopg
from psycopg.rows import dict_row
from flask import Flask, Response, abort, flash, g, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from .ml import FEATURES, LABELS, TARGET, demo_frame, predict_score, risk_level, train_models, validate_training_frame


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
 id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL UNIQUE,
 password_hash TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS students (
 id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 student_number TEXT NOT NULL, name TEXT NOT NULL, program TEXT NOT NULL,
 attendance REAL NOT NULL, quiz_score REAL NOT NULL, assignment_score REAL NOT NULL,
 midterm_score REAL NOT NULL, previous_score REAL NOT NULL,
 actual_final_score REAL, is_demo INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
 UNIQUE(user_id, student_number)
);
CREATE TABLE IF NOT EXISTS training_samples (
 id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 attendance REAL, quiz_score REAL, assignment_score REAL, midterm_score REAL,
 previous_score REAL, final_score REAL NOT NULL, source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_runs (
 user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
 selected_model TEXT NOT NULL, metrics_json TEXT NOT NULL,
 sample_count INTEGER NOT NULL, test_count INTEGER NOT NULL,
 data_source TEXT NOT NULL, trained_at TEXT NOT NULL, model_blob BLOB
);
"""

POSTGRES_SCHEMA = (SCHEMA.replace(" id INTEGER PRIMARY KEY", " id BIGSERIAL PRIMARY KEY")
                   .replace("user_id INTEGER NOT NULL", "user_id BIGINT NOT NULL")
                   .replace("user_id INTEGER PRIMARY KEY", "user_id BIGINT PRIMARY KEY")
                   .replace("model_blob BLOB", "model_blob BYTEA"))
INTEGRITY_ERRORS = (sqlite3.IntegrityError, psycopg.IntegrityError)


class PostgresConnection:
    """Small compatibility layer for the app's parameterized SQLite queries."""

    def __init__(self, url: str):
        self.connection = psycopg.connect(url, row_factory=dict_row)

    def execute(self, statement: str, parameters=()):
        return self.connection.execute(statement.replace("?", "%s"), parameters)

    def executemany(self, statement: str, parameters):
        with self.connection.cursor() as cursor:
            cursor.executemany(statement.replace("?", "%s"), parameters)

    def commit(self):
        self.connection.commit()

    def rollback(self):
        self.connection.rollback()

    def close(self):
        self.connection.close()


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__, instance_relative_config=True)
    if test_config and test_config.get("INSTANCE_PATH"):
        app.instance_path = str(test_config["INSTANCE_PATH"])
    elif os.environ.get("VERCEL"):
        app.instance_path = "/tmp/spps-instance"
    instance = Path(app.instance_path)
    instance.mkdir(parents=True, exist_ok=True)
    key_file = instance / "secret_key"
    if not os.environ.get("SECRET_KEY") and not key_file.exists():
        key_file.write_text(secrets.token_hex(32))
        try:
            key_file.chmod(0o600)
        except OSError:
            pass
    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY") or key_file.read_text(),
        DATABASE=str(instance / "student_performance.sqlite3"),
        DATABASE_URL=os.environ.get("DATABASE_URL"),
        MAX_CONTENT_LENGTH=2 * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=bool(os.environ.get("DATABASE_URL")),
    )
    if test_config:
        app.config.update(test_config)

    def db():
        if "db" not in g:
            if app.config["DATABASE_URL"]:
                connection = PostgresConnection(app.config["DATABASE_URL"])
            else:
                connection = sqlite3.connect(app.config["DATABASE"])
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA foreign_keys = ON")
            g.db = connection
        return g.db

    with app.app_context():
        if app.config["DATABASE_URL"]:
            conn = PostgresConnection(app.config["DATABASE_URL"])
            for statement in POSTGRES_SCHEMA.split(";"):
                if statement.strip():
                    conn.execute(statement)
            conn.commit()
            conn.close()
        else:
            conn = sqlite3.connect(app.config["DATABASE"])
            conn.executescript(SCHEMA)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(model_runs)")}
            if "model_blob" not in columns:
                conn.execute("ALTER TABLE model_runs ADD COLUMN model_blob BLOB")
            conn.commit()
            conn.close()

    @app.teardown_appcontext
    def close_db(_error):
        connection = g.pop("db", None)
        if connection is not None:
            connection.close()

    @app.before_request
    def load_user_and_check_csrf():
        g.user = None
        if session.get("user_id"):
            g.user = db().execute("SELECT id, name, email FROM users WHERE id = ?", (session["user_id"],)).fetchone()
            if g.user is None:
                session.clear()
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            token = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token", "")
            expected = session.get("csrf_token", "")
            if not expected or not secrets.compare_digest(token, expected):
                abort(400, "Invalid CSRF token. Refresh the page and try again.")

    @app.context_processor
    def shared_template_values():
        if "csrf_token" not in session:
            session["csrf_token"] = secrets.token_urlsafe(32)
        return {"csrf_token": session["csrf_token"], "current_user": g.get("user"), "risk_level": risk_level}

    def login_required(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if g.user is None:
                return redirect(url_for("login"))
            return view(*args, **kwargs)
        return wrapped

    def get_model():
        row = db().execute("SELECT model_blob FROM model_runs WHERE user_id = ?", (g.user["id"],)).fetchone()
        return joblib.load(io.BytesIO(bytes(row["model_blob"]))) if row and row["model_blob"] else None

    def model_run():
        row = db().execute(
            "SELECT user_id,selected_model,metrics_json,sample_count,test_count,data_source,trained_at FROM model_runs WHERE user_id = ?",
            (g.user["id"],),
        ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["metrics"] = json.loads(data.pop("metrics_json"))
        return data

    def train_and_save(user_id: int, frame: pd.DataFrame, source: str):
        model, metrics, selected, test_count = train_models(frame)
        model_bytes = io.BytesIO()
        joblib.dump(model, model_bytes)
        db().execute(
            """INSERT INTO model_runs(user_id,selected_model,metrics_json,sample_count,test_count,data_source,trained_at,model_blob)
               VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET
               selected_model=excluded.selected_model, metrics_json=excluded.metrics_json,
               sample_count=excluded.sample_count, test_count=excluded.test_count,
               data_source=excluded.data_source, trained_at=excluded.trained_at,
               model_blob=excluded.model_blob""",
            (user_id, selected, json.dumps(metrics), len(frame), test_count, source, now(), model_bytes.getvalue()),
        )
        db().commit()

    def seed_demo(user_id: int):
        frame = demo_frame()
        db().executemany(
            """INSERT INTO training_samples(user_id,attendance,quiz_score,assignment_score,midterm_score,previous_score,final_score,source)
               VALUES(?,?,?,?,?,?,?,?)""",
            [(user_id, *row, "demo") for row in frame.itertuples(index=False, name=None)],
        )
        examples = [
            ("ST-101", "Ayesha Khan", "Computer Science", 94, 88, 91, 86, 84),
            ("ST-102", "Bilal Ahmed", "Software Engineering", 67, 60, 70, 58, 62),
            ("ST-103", "Fatima Ali", "Computer Science", 48, 40, 42, 36, 51),
            ("ST-104", "Hamza Malik", "Information Technology", 82, 74, 80, 77, 72),
            ("ST-105", "Maryam Raza", "Software Engineering", 96, 90, 92, 93, 88),
            ("ST-106", "Omar Farooq", "Information Technology", 56, 52, 48, 43, 58),
        ]
        db().executemany(
            """INSERT INTO students(user_id,student_number,name,program,attendance,quiz_score,assignment_score,midterm_score,previous_score,is_demo,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            [(user_id, *row, 1, now()) for row in examples],
        )
        train_and_save(user_id, frame, "demo")

    def own_student(student_id: int):
        row = db().execute("SELECT * FROM students WHERE id = ? AND user_id = ?", (student_id, g.user["id"])).fetchone()
        if row is None:
            abort(404)
        return row

    def student_payload(form):
        number = form.get("student_number", "").strip()
        name = form.get("name", "").strip()
        program = form.get("program", "").strip()
        if not 1 <= len(number) <= 40 or not 1 <= len(name) <= 100 or not 1 <= len(program) <= 100:
            raise ValueError("Student ID, name, and program are required (maximum 40, 100, and 100 characters).")
        values = {"student_number": number, "name": name, "program": program}
        for field in FEATURES + ["actual_final_score"]:
            raw = form.get(field, "").strip()
            if field == "actual_final_score" and not raw:
                values[field] = None
                continue
            try:
                value = float(raw)
            except ValueError:
                raise ValueError(f"{field.replace('_', ' ').title()} must be a number between 0 and 100.") from None
            if not 0 <= value <= 100:
                raise ValueError(f"{field.replace('_', ' ').title()} must be between 0 and 100.")
            values[field] = value
        return values

    def student_results():
        students = db().execute("SELECT * FROM students WHERE user_id = ? ORDER BY name", (g.user["id"],)).fetchall()
        model = get_model()
        results = []
        for student in students:
            item = dict(student)
            item["predicted_score"] = predict_score(model, item) if model is not None else None
            item["risk"] = risk_level(item["predicted_score"]) if item["predicted_score"] is not None else "Untrained"
            results.append(item)
        return results

    @app.get("/")
    def index():
        return redirect(url_for("dashboard")) if g.user else render_template("landing.html")

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if g.user:
            return redirect(url_for("dashboard"))
        if request.method == "POST":
            name = request.form.get("name", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            if not 1 <= len(name) <= 100 or "@" not in email or len(email) > 254 or len(password) < 8:
                flash("Enter a name, a valid email, and a password of at least 8 characters.", "error")
            else:
                try:
                    cursor = db().execute(
                        "INSERT INTO users(name,email,password_hash,created_at) VALUES(?,?,?,?) RETURNING id",
                        (name, email, generate_password_hash(password), now()),
                    )
                    user_id = cursor.fetchone()["id"]
                    seed_demo(user_id)
                    session.clear()
                    session["user_id"] = user_id
                    session["csrf_token"] = secrets.token_urlsafe(32)
                    flash("Your workspace is ready with example students and demonstration data.", "success")
                    return redirect(url_for("dashboard"))
                except INTEGRITY_ERRORS:
                    db().rollback()
                    flash("An account with that email already exists.", "error")
        return render_template("auth.html", mode="register")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if g.user:
            return redirect(url_for("dashboard"))
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            row = db().execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
            if row and check_password_hash(row["password_hash"], request.form.get("password", "")):
                session.clear()
                session["user_id"] = row["id"]
                session["csrf_token"] = secrets.token_urlsafe(32)
                return redirect(url_for("dashboard"))
            flash("Email or password is incorrect.", "error")
        return render_template("auth.html", mode="login")

    @app.post("/logout")
    @login_required
    def logout():
        session.clear()
        return redirect(url_for("index"))

    @app.get("/dashboard")
    @login_required
    def dashboard():
        students = student_results()
        counts = {label: sum(s["risk"] == label for s in students) for label in LABELS}
        avg = round(sum(s["predicted_score"] for s in students if s["predicted_score"] is not None) / len(students), 1) if students else 0
        return render_template("dashboard.html", students=students, counts=counts, average=avg, run=model_run())

    @app.get("/students")
    @login_required
    def students():
        query = request.args.get("q", "").strip().lower()
        rows = student_results()
        if query:
            rows = [row for row in rows if query in row["name"].lower() or query in row["student_number"].lower() or query in row["program"].lower()]
        return render_template("students.html", students=rows, query=query, run=model_run())

    @app.route("/students/new", methods=["GET", "POST"])
    @login_required
    def add_student():
        if request.method == "POST":
            try:
                values = student_payload(request.form)
                fields = ["student_number", "name", "program", *FEATURES, "actual_final_score"]
                cursor = db().execute(
                    f"INSERT INTO students(user_id,{','.join(fields)},created_at) VALUES({','.join('?' for _ in range(11))}) RETURNING id",
                    (g.user["id"], *(values[key] for key in fields), now()),
                )
                student_id = cursor.fetchone()["id"]
                db().commit()
                flash("Student added.", "success")
                return redirect(url_for("student_detail", student_id=student_id))
            except (ValueError, *INTEGRITY_ERRORS) as error:
                db().rollback()
                flash("This student ID already exists." if isinstance(error, INTEGRITY_ERRORS) else str(error), "error")
        return render_template("student_form.html", student=request.form if request.method == "POST" else {}, editing=False)

    @app.get("/students/<int:student_id>")
    @login_required
    def student_detail(student_id):
        student = dict(own_student(student_id))
        model = get_model()
        score = predict_score(model, student) if model is not None else None
        return render_template("student_detail.html", student=student, score=score, risk=risk_level(score) if score is not None else None, run=model_run())

    @app.route("/students/<int:student_id>/edit", methods=["GET", "POST"])
    @login_required
    def edit_student(student_id):
        student = own_student(student_id)
        if request.method == "POST":
            try:
                values = student_payload(request.form)
                fields = ["student_number", "name", "program", *FEATURES, "actual_final_score"]
                db().execute(
                    f"UPDATE students SET {','.join(f'{key}=?' for key in fields)},is_demo=0 WHERE id=? AND user_id=?",
                    (*(values[key] for key in fields), student_id, g.user["id"]),
                )
                db().commit()
                flash("Student record updated.", "success")
                return redirect(url_for("student_detail", student_id=student_id))
            except (ValueError, *INTEGRITY_ERRORS) as error:
                db().rollback()
                flash("This student ID already exists." if isinstance(error, INTEGRITY_ERRORS) else str(error), "error")
        return render_template("student_form.html", student=request.form if request.method == "POST" else student, editing=True)

    @app.post("/students/<int:student_id>/delete")
    @login_required
    def delete_student(student_id):
        own_student(student_id)
        db().execute("DELETE FROM students WHERE id = ? AND user_id = ?", (student_id, g.user["id"]))
        db().commit()
        flash("Student deleted.", "success")
        return redirect(url_for("students"))

    @app.get("/training")
    @login_required
    def training():
        run = model_run()
        return render_template("training.html", run=run, labels=LABELS)

    @app.get("/training/chart.png")
    @login_required
    def training_chart():
        run = model_run()
        if run is None:
            abort(404)
        from matplotlib.figure import Figure
        names = [row["name"] for row in run["metrics"]]
        values = [row["mae"] for row in run["metrics"]]
        colors = ["#6bc8ac" if name == run["selected_model"] else "#a5b8dd" for name in names]
        figure = Figure(figsize=(8, 3.4), dpi=130, facecolor="#ffffff")
        axis = figure.subplots()
        axis.barh(names[::-1], values[::-1], color=colors[::-1], height=0.55)
        axis.set_xlabel("Mean absolute error on unseen records (lower is better)", color="#687991", fontsize=9)
        axis.tick_params(axis="both", colors="#687991", labelsize=9, length=0)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#e5eaf1")
        axis.set_axisbelow(True)
        axis.grid(axis="x", color="#edf1f6")
        for index, value in enumerate(values[::-1]):
            axis.text(value + max(values) * 0.025, index, f"{value:.2f}", va="center", color="#334560", fontsize=9)
        axis.set_xlim(0, max(values) * 1.18 if max(values) else 1)
        figure.tight_layout()
        output = io.BytesIO()
        figure.savefig(output, format="png", bbox_inches="tight")
        return Response(output.getvalue(), mimetype="image/png", headers={"Cache-Control": "private, no-store"})

    @app.post("/training/upload")
    @login_required
    def upload_training():
        upload = request.files.get("file")
        if upload is None or not upload.filename.lower().endswith(".csv"):
            flash("Choose a CSV file with the required columns.", "error")
            return redirect(url_for("training"))
        try:
            frame = pd.read_csv(upload.stream)
            frame = validate_training_frame(frame)
            # Train and validate before replacing a working dataset.
            model, metrics, selected, test_count = train_models(frame)
            db().execute("DELETE FROM training_samples WHERE user_id = ?", (g.user["id"],))
            db().execute("DELETE FROM students WHERE user_id = ? AND is_demo = 1", (g.user["id"],))
            db().executemany(
                """INSERT INTO training_samples(user_id,attendance,quiz_score,assignment_score,midterm_score,previous_score,final_score,source)
                   VALUES(?,?,?,?,?,?,?,?)""",
                [(g.user["id"], *(None if pd.isna(v) else float(v) for v in row), "uploaded")
                 for row in frame.itertuples(index=False, name=None)],
            )
            model_bytes = io.BytesIO()
            joblib.dump(model, model_bytes)
            db().execute(
                """UPDATE model_runs SET selected_model=?,metrics_json=?,sample_count=?,test_count=?,data_source=?,trained_at=?,model_blob=? WHERE user_id=?""",
                (selected, json.dumps(metrics), len(frame), test_count, "uploaded", now(), model_bytes.getvalue(), g.user["id"]),
            )
            db().commit()
            flash(f"Model trained on {len(frame)} uploaded historical records.", "success")
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
            db().rollback()
            flash(f"Could not use the CSV: {error}", "error")
        return redirect(url_for("training"))

    @app.post("/training/retrain")
    @login_required
    def retrain():
        samples = db().execute(
            "SELECT attendance,quiz_score,assignment_score,midterm_score,previous_score,final_score FROM training_samples WHERE user_id = ?",
            (g.user["id"],),
        ).fetchall()
        completed = db().execute(
            """SELECT attendance,quiz_score,assignment_score,midterm_score,previous_score,actual_final_score AS final_score
               FROM students WHERE user_id = ? AND actual_final_score IS NOT NULL""",
            (g.user["id"],),
        ).fetchall()
        frame = pd.DataFrame([[row[column] for column in FEATURES + [TARGET]] for row in [*samples, *completed]], columns=FEATURES + [TARGET])
        try:
            source = "demo + completed" if model_run() and model_run()["data_source"].startswith("demo") else "uploaded + completed"
            train_and_save(g.user["id"], frame, source if completed else (model_run()["data_source"] if model_run() else "uploaded"))
            flash(f"Model retrained on {len(frame)} historical records.", "success")
        except ValueError as error:
            flash(str(error), "error")
        return redirect(url_for("training"))

    @app.get("/sample.csv")
    @login_required
    def sample_csv():
        sample = demo_frame(40).to_csv(index=False)
        return Response(sample, mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=example_training_data.csv"})

    @app.get("/reports")
    @login_required
    def reports():
        rows = student_results()
        return render_template("reports.html", students=rows, run=model_run(), counts={label: sum(s["risk"] == label for s in rows) for label in LABELS})

    @app.get("/reports.csv")
    @login_required
    def reports_csv():
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Student ID", "Name", "Program", "Attendance", "Quiz", "Assignment", "Midterm", "Previous", "Predicted Final Score", "Risk Level", "Actual Final Score"])
        for row in student_results():
            def safe(value):
                text = "" if value is None else str(value)
                return "'" + text if text.startswith(("=", "+", "-", "@", "\t", "\r")) else text
            writer.writerow([safe(row["student_number"]), safe(row["name"]), safe(row["program"]),
                             row["attendance"], row["quiz_score"], row["assignment_score"], row["midterm_score"],
                             row["previous_score"], row["predicted_score"], row["risk"], row["actual_final_score"]])
        return Response(output.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=student_performance_report.csv"})

    @app.get("/health")
    def health():
        db().execute("SELECT 1").fetchone()
        return {"status": "ok"}

    @app.errorhandler(413)
    def too_large(_error):
        flash("CSV files must be smaller than 2 MB.", "error")
        return redirect(url_for("training"))

    return app
