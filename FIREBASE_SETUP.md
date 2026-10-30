# Student Performance Prediction System — Firebase Spark (Free) deployment

**Firebase Project ID:** `student-predictor-zain`

**Status:** This branch is an independent JavaScript/browser implementation and is
**not** a literal deployment of the original Flask/scikit-learn project on `main`.

## What this branch provides

- Teacher signup/sign-in using Firebase Authentication **Email/Password**
- Per-teacher student CRUD using Cloud Firestore with strict ownership rules
- Historical training CSV upload (5 predictors and known `final_score`)
- Browser-based Linear Regression, Decision Tree and Random Forest (60 trees)
- Deterministic 60/20/20 split; validation-MAE-based model selection
- Held-out MAE, RMSE, R² and risk-category accuracy
- Browser-side predictions, dashboard and CSV report download
- Offline demonstration mode using browser localStorage (not Firebase storage)
- Synthetic sample dataset, explicitly identified as a demonstration

**Differences from the original:** No Python Flask runtime, no scikit-learn
execution, no server-side model persistence or training, no admin console, and
the mathematical algorithms are independent JavaScript implementations.
Results will differ from the original Python project.

## 1. Create a Firebase project (no payment card)

1. Open https://console.firebase.google.com/
2. Create a project and retain the **Spark (No-cost)** plan.
3. In **Build > Authentication > Sign-in method**, enable **Email/Password**.
4. In **Build > Firestore Database**, create a database in your chosen region.
   Start in production mode (not test mode); deploy the rules below before
   storing user data.
5. In **Project settings > Your apps**, register a **Web app**.
6. Copy its PUBLIC web configuration values into:
   `firebase-public/firebase-config.js`. The project ID is already set to
   `student-predictor-zain`, but you still must replace the example API key,
   messaging sender ID, app ID, and verify the other web app settings.
   These are browser connection identifiers, not private service credentials.
   Do not add a service-account JSON, admin key, or API token to this file.
7. Under **Authentication > Settings > Authorized domains**, verify that
   `student-predictor-zain.web.app` and any custom domain are permitted.

## 2. Deploy Firebase Hosting + Firestore rules

The source for the website is the `firebase-public/` directory.

Using a PC with Node.js and Git installed:

```bash
git clone https://github.com/zainnaqvi512/student-performance-prediction-system.git
cd student-performance-prediction-system
git switch firebase/spark-frontend
npm install -g firebase-tools
firebase login
firebase deploy --only hosting,firestore --project student-predictor-zain
```

Alternatively, download the source archive for this branch, unzip it,
configure `firebase-public/firebase-config.js`, then run the last three
commands inside the unpacked directory. Keep the project on Spark; enabling
billing is NOT required for Hosting/Auth/Firestore within their free quotas.

**Security:** Do not share CLI access tokens or login credentials in chat.
Deploying rules is essential before registering real user accounts.
Firestore rules permit only authenticated owners to access their own records.

**Caution:** A live URL is assigned by Firebase; we cannot infer it until
Hosting is deployed. Default Hosting domain is usually
`https://student-predictor-zain.web.app` (only after successfully deploying). Verify it in the Firebase Console.

## 3. First use

1. Open the site, choose Create account and register a teacher.
2. Add a student using the Students screen; check the prediction and risk.
3. Upload historical CSV from Model training; at least 30 labeled rows.
4. Click Train three algorithms, compare validation MAE and held-out test scores.
5. Export the student prediction report from Reports.
6. Reload the site to verify your student data still exists in Firestore.

Before the Firebase configuration is entered, **Explore offline demonstration**
works without an account. Its data remains only in that device's browser
localStorage and should never contain personal student data.

## CSV format

```csv
attendance,quiz_score,assignment_score,midterm_score,previous_score,final_score
90,84,88,79,82,85
62,55,59,48,60,53
```

Minimum 30 rows. Input scores 0..100; empty predictor cells are imputed
using training-split medians. The target `final_score` must always exist.

## Important limitations

- Training happens **inside the browser**, not in Firebase Functions.
- Firestore stores the historical CSV rows in one document. Avoid importing
  large datasets near Firestore's 1 MiB document-size limit; the UI caps
  imports at 750 KB before serialization.
- The browser retrains models on each login. This is intended for small data
  and teaching demos, not large-scale training or clinical/high-stakes use.
- Default performance metrics from synthetic data demonstrate mechanics only.
- This is a **teacher dashboard only**. Student and administrator logins from
  the original design are not implemented here.
- Risk bands below 50 / 50 to below 80 / 80+ are project defaults, not
  institution-calibrated intervention thresholds.
- Authentication and Firestore free-tier quotas can change; review the
  Firebase pricing page before loading production data.
- The Firebase free tier does not host the original Python Flask backend;
  that requires a separate Python host or a billing-enabled Google backend.

## Tests

Run `node --test tests/firebase-ml.test.mjs` after installing Node.js 20+.
The dedicated GitHub Actions workflow runs these checks on the Firebase branch.

## Files

- `firebase-public/index.html`: website shell
- `firebase-public/styles.css`: responsive interface
- `firebase-public/app.js`: Firebase Auth/Firestore & UI controller
- `firebase-public/ml.mjs`: machine-learning algorithms
- `firebase-public/firebase-config.js`: configure your project identifiers
- `firebase.json`: Hosting and rules deployment
- `firestore.rules`: owner-only database security

Official docs:
https://firebase.google.com/docs/hosting/quickstart
https://firebase.google.com/docs/auth/web/password-auth
https://firebase.google.com/docs/firestore/security/get-started
