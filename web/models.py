"""
Database Models — SQLAlchemy models for tracking application data.
"""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class PredictionHistory(db.Model):
    """
    Stores a history of all predictions made by users.
    """
    __tablename__ = 'prediction_history'

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Inputs
    age = db.Column(db.Integer, nullable=False)
    workclass = db.Column(db.Integer, nullable=False)
    education_num = db.Column(db.Integer, nullable=False)
    marital_status = db.Column(db.Integer, nullable=False)
    occupation = db.Column(db.Integer, nullable=False)
    relationship = db.Column(db.Integer, nullable=False)
    race = db.Column(db.Integer, nullable=False)
    sex = db.Column(db.Integer, nullable=False)
    capital_gain = db.Column(db.Integer, nullable=False)
    capital_loss = db.Column(db.Integer, nullable=False)
    hours_per_week = db.Column(db.Integer, nullable=False)
    native_country = db.Column(db.Integer, nullable=False)
    
    # Outputs
    prediction_result = db.Column(db.Integer, nullable=False) # 0 or 1
    confidence_score = db.Column(db.Float, nullable=True)     # 0.0 to 1.0

    def to_dict(self):
        """Convert model to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() + "Z",
            "inputs": {
                "age": self.age,
                "workclass": self.workclass,
                "education_num": self.education_num,
                "marital_status": self.marital_status,
                "occupation": self.occupation,
                "relationship": self.relationship,
                "race": self.race,
                "sex": self.sex,
                "capital_gain": self.capital_gain,
                "capital_loss": self.capital_loss,
                "hours_per_week": self.hours_per_week,
                "native_country": self.native_country,
            },
            "outputs": {
                "prediction": self.prediction_result,
                "confidence": self.confidence_score,
            }
        }
