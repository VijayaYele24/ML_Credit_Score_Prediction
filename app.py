
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = BASE_DIR / "models" / "credit_score_artifacts.joblib"

# Load the trained model and preprocessing artifacts once.
artifacts = joblib.load(ARTIFACT_PATH)

model = artifacts["model"]
bin_edges = artifacts["bin_edges"]
woe_mappings = artifacts["woe_mappings"]
feature_names = artifacts["feature_names"]
numerical_columns = artifacts["numerical_columns"]


def transform_input(customer_data):
    """Apply the saved training-time WOE transformations."""

    raw = pd.DataFrame([customer_data])
    transformed = pd.DataFrame(index=raw.index)

    for column in feature_names:
        value = float(raw.iloc[0][column])

        if column not in numerical_columns:
            raise ValueError(
                f"Missing numeric preprocessing for {column}"
            )

        edges = bin_edges[column]

        # Match the bins used during model training.
        interval = pd.cut(
            pd.Series([value]),
            bins=edges,
            include_lowest=True
        ).iloc[0]

        # Use the saved WOE mapping.
        # Unseen/out-of-range bins fall back to zero,
        # consistent with the notebook's test transformation.
        woe_value = woe_mappings[column].get(interval, 0.0)

        transformed[column] = [woe_value]

    # Preserve exactly the model's training feature order.
    return transformed[feature_names].astype(float)


def build_result(probability):
    credit_score = int(
        np.clip(900 - probability * 600, 300, 900)
    )

    if credit_score >= 750:
        risk = "Low Risk"
        decision = "Approve"
        credit_limit = 50000
        pricing = "Low Interest Rate"

    elif credit_score >= 600:
        risk = "Medium Risk"
        decision = "Manual Review"
        credit_limit = 25000
        pricing = "Medium Interest Rate"

    else:
        risk = "High Risk"
        decision = "Reject"
        credit_limit = 0
        pricing = "High Interest Rate"

    return {
        "pd": round(probability * 100, 2),
        "credit_score": credit_score,
        "risk": risk,
        "decision": decision,
        "credit_limit": credit_limit,
        "pricing": pricing
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    values = {}

    if request.method == "POST":
        try:
            customer_data = {}

            for column in feature_names:
                raw_value = request.form.get(column, "").strip()

                if not raw_value:
                    raise ValueError(
                        f"Please enter a value for {column}."
                    )

                value = float(raw_value)

                if not np.isfinite(value):
                    raise ValueError(
                        f"{column} must be a finite number."
                    )

                customer_data[column] = value
                values[column] = raw_value

            transformed = transform_input(customer_data)

            # Find the probability for class 1 (default).
            classes = list(model.classes_)
            default_index = classes.index(1)

            probability = model.predict_proba(
                transformed
            )[0, default_index]

            result = build_result(float(probability))

        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)

    return render_template(
        "index.html",
        features=feature_names,
        values=values,
        result=result,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
