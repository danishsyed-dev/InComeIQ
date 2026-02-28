"""
Vercel entrypoint — re-exports the Flask app from the factory.

Vercel's @vercel/python runtime imports this file and looks for
an object named `app` that is a WSGI callable.
"""

import sys
import os

# Ensure the project root is on the path so `web`, `pipelines`, etc. resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.app import create_app

app = create_app()
