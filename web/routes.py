"""
Route Handlers — Flask Blueprint with prediction routes.

GET  / → Render the input form
POST / → Validate input, run prediction, render results
"""

from flask import Blueprint, render_template, request

from pipelines.predict import PredictionPipeline, CustomInput
from web.validators import validate_prediction_input
from web.models import db, PredictionHistory

main_bp = Blueprint("main", __name__)

# Initialize prediction pipeline once (model is cached after first call)
prediction_pipeline = PredictionPipeline()


@main_bp.route("/", methods=["GET", "POST"])
def predict():
    """Handle the main prediction form."""
    if request.method == "GET":
        return render_template("home.html")

    # Validate input
    parsed, error = validate_prediction_input(request.form)
    if error:
        return render_template("home.html", error=error)

    # Build input DataFrame
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

    # Run prediction
    result, probability, plot_filename = prediction_pipeline.predict(features_df)
    
    # Save to history
    try:
        history_entry = PredictionHistory(
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
            prediction_result=result,
            confidence_score=probability,
        )
        db.session.add(history_entry)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"Failed to save prediction history: {e}")

    if result == 1:
        message = "Your Yearly Income is Predicted to be More Than $50,000"
        result_class = "positive"
    else:
        message = "Your Yearly Income is Predicted to be $50,000 or Less"
        result_class = "negative"

    # Format probability if available
    prob_str = f"{probability * 100:.1f}%" if probability is not None else "N/A"

    return render_template(
        "results.html",
        prediction=message,
        result_class=result_class,
        probability=prob_str,
        plot_filename=plot_filename,
    )
