"""
Flask Application Factory — Creates and configures the Flask app.

Uses the factory pattern for testability and flexibility.
"""

from flask import Flask


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

    # Register routes blueprint
    from web.routes import main_bp
    app.register_blueprint(main_bp)

    return app
