"""
REST API — Endpoints for programmatic access.

Exposes endpoints for:
- POST /api/predict: Submit JSON data to get a prediction
- GET  /api/history: Retrieve past predictions from the database
"""

from flask import Blueprint, request, jsonify
from pipelines.predict import PredictionPipeline, CustomInput
from web.validators import validate_prediction_input
from web.models import db, PredictionHistory

api_bp = Blueprint("api", __name__)

# Initialize pipeline once for API blueprint
prediction_pipeline = PredictionPipeline()


@api_bp.route("/predict", methods=["POST"])
def predict():
    """
    API endpoint to make programmatic predictions.
    Expects a JSON payload with the 12 feature fields.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    json_data = request.get_json()
    
    # Validation uses the exact same logic as the UI
    parsed, error = validate_prediction_input(json_data)
    if error:
        return jsonify({"error": error}), 400

    try:
        user_input = CustomInput(
            age=parsed["age"],
            workclass=parsed["workclass"],
            education_num=parsed["education_num"],
            marital_status=parsed["marital_status"],
            occupation=parsed["occupation"],
            relationship=parsed["relationship"],
            race=parsed["race"],
            sex=parsed["sex"],
            capital_gain=parsed["capital_gain"],
            capital_loss=parsed["capital_loss"],
            hours_per_week=parsed["hours_per_week"],
            native_country=parsed["native_country"],
        )
        
        features_df = user_input.to_dataframe()
        
        result, probability, _ = prediction_pipeline.predict(features_df)
        
        # Save to database
        history_entry = PredictionHistory(
            **parsed,
            prediction_result=result,
            confidence_score=probability
        )
        db.session.add(history_entry)
        db.session.commit()

        return jsonify({
            "status": "success",
            "prediction": result,
            "prediction_label": ">50K" if result == 1 else "<=50K",
            "confidence": probability,
            "history_id": history_entry.id
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@api_bp.route("/history", methods=["GET"])
def get_history():
    """
    Retrieve prediction history from the database.
    Supports ?limit=N query parameter (default 10).
    """
    try:
        limit = request.args.get("limit", 10, type=int)
        
        history = PredictionHistory.query.order_by(
            PredictionHistory.created_at.desc()
        ).limit(limit).all()
        
        return jsonify({
            "status": "success",
            "count": len(history),
            "data": [entry.to_dict() for entry in history]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
