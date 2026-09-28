import io

import pytest

from spps import create_app
from spps.ml import demo_frame


@pytest.fixture
def app(tmp_path):
    return create_app({
        "TESTING": True,
        "INSTANCE_PATH": tmp_path,
        "DATABASE": str(tmp_path / "test.sqlite3"),
        "DATABASE_URL": None,
        "SECRET_KEY": "test-secret",
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
