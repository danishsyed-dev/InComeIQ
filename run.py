"""
ML Income Predictor — Application Entry Point.

Starts the Flask web server for income predictions.
Usage: python run.py
"""

from web.app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
