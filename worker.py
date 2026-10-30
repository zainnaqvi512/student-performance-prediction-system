"""Experimental Cloudflare Python Worker entrypoint for the existing Flask app.

This addresses the missing Worker entrypoint. It DOES NOT establish
compatibility of the database, psycopg, SciPy/scikit-learn, or training
operations with the Cloudflare Workers runtime. Test before merging.
"""
import os

# Reuse the original application's temporary instance directory behavior.
# /tmp storage is not durable; use a supported persistent DB adapter before
# storing student records or relying on account/session persistence.
os.environ.setdefault("VERCEL", "1")

from workers import wsgi
from spps import create_app

application = create_app()
Default = wsgi.entrypoint(application)
