import io

import pytest

from spps import create_app
from spps.ml import demo_frame
from werkzeug.security import generate_password_hash


@pytest.fixture
def app(tmp_path):
    return create_app({
        "TESTING": True,
        "INSTANCE_PATH": tmp_path,
        "DATABASE": str(tmp_path / "test.sqlite3"),
        "DATABASE_URL": None,
        "SECRET_KEY": "test-secret",
        "ADMIN_EMAIL": "admin@example.edu",
        "ADMIN_PASSWORD_HASH": generate_password_hash("admin-test-password"),
    })


@pytest.fixture
def client(app):
    return app.test_client()


def csrf(client):
    with client.session_transaction() as state:
        return state["csrf_token"]


def register(client, name="Teacher One", email="teacher@example.edu"):
    client.get("/register")
    return client.post("/register", data={
        "csrf_token": csrf(client), "name": name, "email": email, "password": "securepass123"
    }, follow_redirects=True)


def test_teacher_workflow(client):
    assert b"See potential" in client.get("/").data
    response = register(client)
    assert response.status_code == 200
    assert b"Good to see you" in response.data
    assert b"demonstration data" in response.data

    response = client.get("/training")
    assert b"Model comparison" in response.data
    assert b"Linear Regression" in response.data
    assert b"Decision Tree" in response.data
    assert b"Random Forest" in response.data
    chart = client.get("/training/chart.png")
    assert chart.status_code == 200
    assert chart.data.startswith(b"\x89PNG")

    record = {
        "csrf_token": csrf(client), "student_number": "NEW-001", "name": "New Student",
        "program": "Computer Science", "attendance": "75", "quiz_score": "70",
        "assignment_score": "80", "midterm_score": "65", "previous_score": "72",
        "actual_final_score": "",
    }
    response = client.post("/students/new", data=record, follow_redirects=True)
    assert response.status_code == 200
    assert b"New Student" in response.data
    assert b"PREDICTED OUTCOME" in response.data

    report = client.get("/reports.csv")
    assert report.status_code == 200
    assert b"NEW-001" in report.data
    assert b"Predicted Final Score" in report.data
    assert b"Performance report" in client.get("/reports").data

    with client.application.app_context():
        import sqlite3
        with sqlite3.connect(client.application.config["DATABASE"]) as db:
            student_id = db.execute("SELECT id FROM students WHERE student_number='NEW-001'").fetchone()[0]
    record["name"] = "Updated Student"
    response = client.post(f"/students/{student_id}/edit", data=record, follow_redirects=True)
    assert b"Updated Student" in response.data
    response = client.post(f"/students/{student_id}/delete", data={"csrf_token": csrf(client)}, follow_redirects=True)
    assert response.status_code == 200
    assert b"NEW-001" not in response.data


def test_upload_replaces_demo_data_and_validates_csv(client):
    register(client)
    bad = client.post("/training/upload", data={
        "csrf_token": csrf(client), "file": (io.BytesIO(b"attendance,final_score\n90,80\n"), "bad.csv")
    }, content_type="multipart/form-data", follow_redirects=True)
    assert b"Missing columns" in bad.data
    assert b"synthetic demonstration data" in bad.data

    csv_bytes = demo_frame(65).to_csv(index=False).encode()
    good = client.post("/training/upload", data={
        "csrf_token": csrf(client), "file": (io.BytesIO(csv_bytes), "history.csv")
    }, content_type="multipart/form-data", follow_redirects=True)
    assert good.status_code == 200
    assert b"65 uploaded historical records" in good.data
    assert b"synthetic demonstration data" not in good.data
    assert b"Uploaded" in good.data
    assert b"ST-101" not in client.get("/students").data


def test_auth_csrf_and_account_isolation(app):
    first = app.test_client()
    second = app.test_client()
    register(first, "Teacher A", "a@example.edu")
    register(second, "Teacher B", "b@example.edu")
    assert first.post("/students/new", data={"name": "No token"}).status_code == 400
    with app.app_context():
        import sqlite3
        with sqlite3.connect(app.config["DATABASE"]) as db:
            owned_by_first = db.execute("SELECT id FROM students WHERE student_number='ST-101' AND user_id=1").fetchone()[0]
    assert second.get(f"/students/{owned_by_first}").status_code == 404
    assert second.post(f"/students/{owned_by_first}/delete", data={"csrf_token": csrf(second)}).status_code == 404
    first.post("/logout", data={"csrf_token": csrf(first)})
    assert first.get("/dashboard").status_code == 302
    first.get("/login")
    signed_in = first.post("/login", data={
        "csrf_token": csrf(first), "email": "a@example.edu", "password": "securepass123",
    }, follow_redirects=True)
    assert b"Good to see you, Teacher" in signed_in.data


def test_admin_login_portfolios_and_demo_generation(app):
    teacher = app.test_client()
    admin = app.test_client()
    register(teacher, "Teacher One", "teacher@example.edu")

    assert admin.get("/admin").location.endswith("/admin/login")
    assert teacher.get("/admin").location.endswith("/admin/login")
    admin.get("/admin/login")
    wrong = admin.post("/admin/login", data={
        "csrf_token": csrf(admin), "email": "admin@example.edu", "password": "wrong-password",
    }, follow_redirects=True)
    assert b"incorrect" in wrong.data
    signed_in = admin.post("/admin/login", data={
        "csrf_token": csrf(admin), "email": "admin@example.edu", "password": "admin-test-password",
    }, follow_redirects=True)
    assert b"All workspaces" in signed_in.data
    assert b"Teacher One" in signed_in.data
    assert admin.get("/dashboard").location.endswith("/login")
    assert admin.get("/admin/users/1").status_code == 200
    assert b"Ayesha Khan" in admin.get("/admin/users/1").data
    assert b"ACTUAL FINAL" in admin.get("/admin/users/1").data
    assert b"87.0" in admin.get("/admin/users/1").data

    response = admin.post("/admin/users/1/demo-students", data={"csrf_token": csrf(admin)}, follow_redirects=True)
    assert b"Six sample students were added" in response.data
    assert b"12 students" in response.data
    assert teacher.get("/students").data.count(b"Ayesha Khan") >= 2
    retrained = teacher.post("/training/retrain", data={"csrf_token": csrf(teacher)}, follow_redirects=True)
    assert b"Model retrained on 240 historical records" in retrained.data

    response = admin.post("/admin/demo-workspaces", data={"csrf_token": csrf(admin)}, follow_redirects=True)
    assert b"Demo workspace ready" in response.data
    assert b"Demo Teacher" in response.data
    assert b"demo-" in response.data
    assert b"Demo Teacher" in admin.get("/admin").data
    assert teacher.get("/admin/users/2").location.endswith("/admin/login")

    loaded = admin.post("/admin/load-samples", data={"csrf_token": csrf(admin)}, follow_redirects=True)
    assert b"4 sample workspaces added" in loaded.data
    assert b"Sara Ali" in loaded.data and b"Farhan Ahmed" in loaded.data
    assert b"24" in loaded.data
    again = admin.post("/admin/load-samples", data={"csrf_token": csrf(admin)}, follow_redirects=True)
    assert b"already available" in again.data
    with app.app_context():
        import sqlite3
        with sqlite3.connect(app.config["DATABASE"]) as db:
            sara_id = db.execute("SELECT id FROM users WHERE email='showcase-sara@example.invalid'").fetchone()[0]
    sara = admin.get(f"/admin/users/{sara_id}")
    assert b"Hira Noor" in sara.data and b"91.0" in sara.data

    admin.post("/admin/logout", data={"csrf_token": csrf(admin)})
    assert admin.get("/admin").location.endswith("/admin/login")
