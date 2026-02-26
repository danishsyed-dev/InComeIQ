"""
Flask Application Factory — Creates and configures the Flask app.

Uses the factory pattern for testability and flexibility.
"""

import os
from flask import Flask
from web.models import db


def create_app() -> Flask:
    """
    Create and configure the Flask application.

    Returns:
        Configured Flask app instance.
    """
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    app.config["SECRET_KEY"] = "ml-income-predictor-secret"
    
    # Database config
    db_path = os.path.join(app.root_path, '..', 'data', 'prediction_history.db')
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    
    # Initialize DB
    db.init_app(app)

    # Create tables if they don't exist
    with app.app_context():
        db.create_all()

    # Register blueprints
    from web.routes import main_bp
    from web.api import api_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix='/api')

    return app
