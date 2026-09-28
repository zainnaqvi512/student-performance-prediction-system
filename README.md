# Student Performance Prediction System

A working Flask implementation of the supplied final-year project presentation. Teachers can register, manage student records, train and compare three machine-learning models, review predicted final scores and risk levels, and download a performance report.

Live demo: <https://student-performance-prediction-syst-one.vercel.app/>. It runs on Vercel Hobby with a free Render PostgreSQL database. The current free database expires on **28 October 2026**; export or migrate its data before then to keep accounts and records. Deployment API keys are stored only in the local, ignored `.env.local` file.

## Accounts and admin

Teachers sign up at `/register` and sign in at `/login`. Each teacher sees only their own students, training data, and reports. A new workspace starts with clearly labeled sample students and an illustrative model.

The separate admin sign-in is at `/admin/login`. Admin credentials come from `ADMIN_EMAIL` and `ADMIN_PASSWORD_HASH` environment variables; the admin route is unavailable until both are configured. The admin console lists teacher workspaces, shows each user's student portfolio and prediction summary, and can create a demo teacher workspace or add six labeled sample students to a selected workspace. Sample generation does not replace existing records or retrain the user's model. The generated demo teacher password is displayed to the admin once.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

Open <http://127.0.0.1:5000>, create an account, and explore the example students. Local account data and trained models are stored in SQLite under `instance/`. When `DATABASE_URL` is set, the app uses PostgreSQL instead; the model is stored in the database, so it survives web-service restarts.

## Deploy on Render

`render.yaml` defines a free Python web service and a free Render Postgres database in Singapore. Connect this repository as a Render Blueprint. Render generates the Flask secret key and passes the private database URL to the service. The start command uses Gunicorn. The app can also run as a Vercel Python function using `api/index.py` and `vercel.json`, with `DATABASE_URL` set to a PostgreSQL connection string and `SECRET_KEY` set to a stable random value.

The free web service can sleep when idle, so its first request may be slow. Render's free Postgres database expires after 30 days; upgrade the database or migrate the data before then if you need a lasting deployment. Do not use the free deployment for real student records without an institution-approved data policy.

## Use real historical data

Open **Model training** and upload a CSV with these columns:

```csv
attendance,quiz_score,assignment_score,midterm_score,previous_score,final_score
90,84,88,79,82,85
62,55,59,48,60,53
```

Each score must be a percentage from 0 to 100. Blank input features are imputed using the median from training data; every row needs a known `final_score`. At least 30 historical rows are required. The downloadable example CSV shows the format but contains synthetic data and should not be used to claim real accuracy.

The app compares Linear Regression, Decision Tree, and Random Forest. It selects the model with the lowest validation MAE, then evaluates all three on a separate 20% held-out test split. It reports MAE, RMSE, R², risk-classification accuracy, and confusion matrices, then refits the chosen model on all available historical rows. Risk bands are **At Risk** below 50, **Average** from 50 to below 80, and **Excellent** at 80 or above. These are project defaults; align them with your institution's grading policy before real use.

After final results arrive, add actual final scores to student records and click **Retrain with completed student records**. Uploaded historical rows remain available for retraining.

## Scope and limitations

- New accounts start with synthetic demonstration data and example students so the full workflow can be explored. Demo predictions and metrics are illustrative, not evidence of real-world performance.
- Uploading real historical data removes untouched example students; student records you added or edited are kept.
- The app keeps accounts separate and uses password hashing, CSRF tokens, upload limits, and ownership checks. For a production deployment, use HTTPS, backups, a managed database, and an institution-approved student-data policy.
- Predictions are support signals. Teachers should review individual circumstances and actual outcomes before taking action.

## Tests

```bash
pytest -q
```
