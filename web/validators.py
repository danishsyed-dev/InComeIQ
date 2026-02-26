"""
Input Validators — Sanitize and validate form data.

Provides meaningful error messages instead of raw Python exceptions
when users submit invalid data.
"""

from typing import Dict, Optional, Tuple


def validate_prediction_input(form_data: Dict) -> Tuple[Optional[Dict[str, int]], Optional[str]]:
    """
    Validate and parse the prediction form data.

    Args:
        form_data: Flask request.form dict.

    Returns:
        Tuple of (parsed_data_dict, error_message).
        If validation passes: (dict, None).
        If validation fails: (None, error_string).
    """
    required_fields = [
        "age", "workclass", "education_num", "marital_status",
        "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
    ]

    parsed = {}

    for field in required_fields:
        value = form_data.get(field)

        if value is None or value.strip() == "":
            return None, f"Missing required field: {field}"

        try:
            parsed[field] = int(value)
        except ValueError:
            return None, f"Invalid value for {field}: must be a number"

    # Range validation for key fields
    if parsed["age"] < 0 or parsed["age"] > 120:
        return None, "Age must be between 0 and 120"

    if parsed["hours_per_week"] < 0 or parsed["hours_per_week"] > 168:
        return None, "Hours per week must be between 0 and 168"

    if parsed["capital_gain"] < 0:
        return None, "Capital gain cannot be negative"

    if parsed["capital_loss"] < 0:
        return None, "Capital loss cannot be negative"

    return parsed, None
