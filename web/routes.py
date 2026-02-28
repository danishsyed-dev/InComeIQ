"""
Route Handlers — Flask Blueprint with prediction routes.

GET  / -> Render the input form
POST / -> Validate input, run prediction, render results
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
        message = "Income > $50,000 / year"
        result_class = "positive"
    else:
        message = "Income <= $50,000 / year"
        result_class = "negative"

    # Format probability if available
    prob_str = f"{probability * 100:.1f}%" if probability is not None else "N/A"
    prob_pct = f"{probability * 100:.1f}" if probability is not None else None

    # ── Human-readable label maps ─────────────────────────────────
    workclass_map = {
        0: "Federal Government", 1: "Local Government", 2: "Never Worked",
        3: "Private", 4: "Self-employed (Inc)", 5: "Self-employed (Not Inc)",
        6: "State Government", 7: "Without Pay",
    }
    education_map = {
        1: "Preschool", 2: "1st-4th Grade", 3: "5th-6th Grade", 4: "7th-8th Grade",
        5: "9th Grade", 6: "10th Grade", 7: "11th Grade", 8: "12th Grade",
        9: "High School Grad", 10: "Some College", 11: "Associate (Vocational)",
        12: "Associate (Academic)", 13: "Bachelors", 14: "Masters",
        15: "Prof School", 16: "Doctorate",
    }
    marital_map = {
        0: "Divorced", 1: "Married (Armed Forces)", 2: "Married (Civilian)",
        3: "Married (Spouse Absent)", 4: "Never Married", 5: "Separated", 6: "Widowed",
    }
    occupation_map = {
        0: "Administrative / Clerical", 1: "Armed Forces", 2: "Craft / Repair",
        3: "Executive / Managerial", 4: "Farming / Fishing", 5: "Handlers / Cleaners",
        6: "Machine Operator", 7: "Other Service", 8: "Private Household",
        9: "Professional Specialty", 10: "Protective Services", 11: "Sales",
        12: "Tech Support", 13: "Transport / Moving",
    }
    relationship_map = {
        0: "Husband", 1: "Not in Family", 2: "Other Relative",
        3: "Own Child", 4: "Unmarried", 5: "Wife",
    }
    race_map = {
        0: "American Indian / Eskimo", 1: "Asian / Pacific Islander",
        2: "Black", 3: "Other", 4: "White",
    }
    sex_map = {0: "Female", 1: "Male"}

    # Selected model
    model_name_map = {
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "svm": "SVM",
    }
    selected_model = request.form.get("model", "random_forest")
    model_label = model_name_map.get(selected_model, "Random Forest")

    input_summary = {
        "Age": parsed["age"],
        "Sex": sex_map.get(parsed["sex"], parsed["sex"]),
        "Race": race_map.get(parsed["race"], parsed["race"]),
        "Marital Status": marital_map.get(parsed["marital_status"], parsed["marital_status"]),
        "Relationship": relationship_map.get(parsed["relationship"], parsed["relationship"]),
        "Education": education_map.get(parsed["education_num"], parsed["education_num"]),
        "Employment": workclass_map.get(parsed["workclass"], parsed["workclass"]),
        "Occupation": occupation_map.get(parsed["occupation"], parsed["occupation"]),
        "Hours / Week": parsed["hours_per_week"],
        "Capital Gain": f"${parsed['capital_gain']:,}",
        "Capital Loss": f"${parsed['capital_loss']:,}",
    }

    return render_template(
        "results.html",
        prediction=message,
        result_class=result_class,
        probability=prob_str,
        prob_pct=prob_pct,
        plot_filename=plot_filename,
        input_summary=input_summary,
        model_label=model_label,
    )
