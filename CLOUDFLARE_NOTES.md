# Cloudflare deployment experiment — NOT production-ready

## Why the original build failed

The original repository has Vercel-specific configuration but no Cloudflare
Worker entrypoint. Without a Wrangler configuration file or an index.html
static directory, `npx wrangler deploy` failed with:

> Could not detect a directory containing static files

## What this branch changes

- `wrangler.jsonc` points to `worker.py`.
- `worker.py` exposes the existing Flask application through Cloudflare's
  official Python Workers WSGI bridge.
- `pyproject.toml` lists candidate dependencies for `pywrangler`.

## Test the build — DO NOT merge solely because deployment succeeds

Use Cloudflare's Python Worker command rather than treating this as a static
site. In Workers & Pages > the connected project > Settings > Builds, use:

- Build command: `python -m pip install --user uv` (if uv is unavailable)
- Deploy command: `uv run pywrangler deploy`
- Root directory: `/`

If the build environment cannot execute `uv`, install it in the build
environment and ensure its bin path is available before running pywrangler.

The Python dependencies are NOT proven compatible with the Pyodide runtime.
In particular, psycopg, NumPy/SciPy/scikit-learn bundle size, and startup
resource usage require validation. These may cause another build/runtime error.

## IMPORTANT: persisted data and the Free-plan resource limits

The original app stores student/teacher records in SQLite locally, or a
PostgreSQL URL via psycopg. Merely deploying this Python WSGI entrypoint does
NOT give Cloudflare Workers a durable SQLite filesystem or a compatible
Postgres connection.

Cloudflare Workers Free provides 128 MB per isolate and 10 ms CPU per HTTP
request. Model fitting may exceed these limits. A real production port
must migrate storage to D1 (or another supported persistent backend) and
move CPU-intensive model training outside the Worker.

This is an experimental adapter to get past the Wrangler static-directory
error and expose subsequent compatibility failures. It is not a verified
functional deployment of the original full-stack student prediction app.

Cloudflare docs:
https://developers.cloudflare.com/workers/languages/python/packages/flask/
https://developers.cloudflare.com/workers/languages/python/packages/
https://developers.cloudflare.com/workers/platform/limits/
