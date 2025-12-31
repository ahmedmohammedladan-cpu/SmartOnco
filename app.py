from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import os

app = Flask(__name__)

# ==================================================
# BREAST CANCER MODEL
# ==================================================
selected_features_bc = [
    "worst radius", "mean concave points", "worst perimeter", "mean concavity",
    "worst concave points", "mean radius", "worst area", "mean perimeter",
    "mean texture", "worst smoothness"
]

data_bc = load_breast_cancer()
indices_bc = [list(data_bc.feature_names).index(f) for f in selected_features_bc]
X_bc = data_bc.data[:, indices_bc]
y_bc = data_bc.target

X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(
    X_bc, y_bc, test_size=0.2, random_state=42
)

model_bc = DecisionTreeClassifier(random_state=42)
model_bc.fit(X_train_bc, y_train_bc)

demo_values_bc = X_test_bc[0].tolist()

# ==================================================
# MANUAL RULES – BREAST CANCER
# ==================================================
def predict_breast_manual(features):
    worst_radius = features[0]
    worst_area = features[6]
    worst_concave_pts = features[4]

    if worst_radius > 18 or worst_area > 1000:
        return "Malignant (Cancerous)", "High"

    if worst_radius > 14 and worst_concave_pts > 0.05:
        return "Malignant (Cancerous)", "Moderate"

    if worst_radius < 12 and worst_area < 500:
        return "Benign (Non-Cancerous)", "Low"

    # fallback to ML model
    return "ml_model", None

# ==================================================
# LUNG CANCER – MANUAL RULE SYSTEM
# ==================================================
def predict_lung_manual(features):
    age = int(features[1])
    smoking = int(features[2])
    coughing = int(features[11])
    chest_pain = int(features[14])

    risk_score = 0

    if age > 60:
        risk_score += 3
    if smoking == 2:
        risk_score += 4
    if coughing == 2:
        risk_score += 2
    if chest_pain == 2:
        risk_score += 3

    if risk_score >= 7:
        return "Cancer Detected", "High"
    elif risk_score >= 4:
        return "Suspicious Findings", "Moderate"
    else:
        return "No Cancer Detected", "Low"

# ==================================================
# WEB ROUTES
# ==================================================
def format_result(prediction, risk):
    """
    Returns a professionally formatted multi-line string for display
    """
    if "Malignant" in prediction or "Cancer Detected" in prediction:
        recommendation = "Consult a certified oncologist for further medical evaluation."
    elif "Suspicious" in prediction:
        recommendation = "Seek professional medical consultation for confirmation."
    else:
        recommendation = "Routine medical checkups are advised."

    result = f"Prediction: {prediction}\nRisk Level: {risk}\nRecommendation: {recommendation}"
    return result

@app.route("/")
def home():
    return render_template(
        "index.html",
        features_bc=selected_features_bc,
        demo_values_bc=demo_values_bc
    )

@app.route("/predict_bc", methods=["POST"])
def predict_bc():
    feature_vals = [
        float(request.form.get(f"feature_bc{i+1}"))
        for i in range(len(selected_features_bc))
    ]

    prediction, risk = predict_breast_manual(feature_vals)

    if prediction == "ml_model":
        df = pd.DataFrame([feature_vals], columns=selected_features_bc)
        pred = model_bc.predict(df)[0]
        prediction = "Malignant (Cancerous)" if pred == 0 else "Benign (Non-Cancerous)"
        risk = "Moderate"

    prediction_text_html = format_result(prediction, risk).replace("\n", "<br>")

    return render_template(
        "index.html",
        features_bc=selected_features_bc,
        demo_values_bc=demo_values_bc,
        prediction_text=prediction_text_html
    )

@app.route("/lung")
def lung_page():
    return render_template("lung_cancer.html")

@app.route("/predict_lc", methods=["POST"])
def predict_lc():
    feature_vals = [
        float(request.form.get(f"feature_lc{i+1}", 1))
        for i in range(15)
    ]

    prediction, risk = predict_lung_manual(feature_vals)

    prediction_text_html = format_result(prediction, risk).replace("\n", "<br>")

    return render_template(
        "lung_cancer.html",
        prediction_text=prediction_text_html
    )

@app.route("/prostate")
def prostate_page():
    return render_template("prostate.html")

@app.route("/predict_prostate", methods=["POST"])
def predict_prostate():
    psa = float(request.form.get("feature_prostate2", 0))
    biopsy = int(request.form.get("feature_prostate3", 0))

    if biopsy == 1 or psa > 10:
        prediction = "Prostate Cancer Detected"
        risk = "High"
    else:
        prediction = "No Prostate Cancer Detected"
        risk = "Low"

    prediction_text_html = format_result(prediction, risk).replace("\n", "<br>")

    return render_template(
        "prostate.html",
        prediction_text=prediction_text_html
    )

# ==================================================
# API ROUTE
# ==================================================
@app.route("/api/predict/breast", methods=["POST"])
def api_predict_breast():
    try:
        data = request.get_json()
        feature_vals = data.get("features", [])

        if len(feature_vals) != len(selected_features_bc):
            return jsonify({"error": "Invalid feature count"}), 400

        prediction, risk = predict_breast_manual(feature_vals)

        if prediction == "ml_model":
            df = pd.DataFrame([feature_vals], columns=selected_features_bc)
            pred = model_bc.predict(df)[0]
            prediction = "Malignant (Cancerous)" if pred == 0 else "Benign (Non-Cancerous)"
            risk = "Moderate"

        full_result = format_result(prediction, risk)

        response = {
            "prediction": prediction,
            "risk_level": risk,
            "recommendation": full_result.split("\n")[2],  # only recommendation
            "full_result": full_result,
            "disclaimer": (
                "This system is for decision support only "
                "and does not replace professional medical diagnosis."
            )
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": "Invalid request", "details": str(e)}), 500

# ==================================================
# RUN APP
# ==================================================
import os 
if __name__ == "__main__": 
    port = int(os.environ.get("PORT", 5000)) 
    app.run(host="0.0.0.0", port=port, debug=True)

