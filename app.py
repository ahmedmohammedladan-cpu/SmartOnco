from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# =========================
# Gemini AI Import (Latest SDK)
# =========================
from google import genai
from google.genai.types import GenerateContentConfig

# =========================
# Initialize Flask
# =========================
app = Flask(__name__)

# =========================
# Gemini AI Client Setup
# =========================
GENAI_KEY = os.environ.get("GENAI_API_KEY", "")
client = genai.Client(api_key=GENAI_KEY) if GENAI_KEY else None

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

# Initialize clustering for breast cancer
scaler_bc = StandardScaler()
X_scaled_bc = scaler_bc.fit_transform(X_bc)
kmeans_bc = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_bc.fit(X_scaled_bc)

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
    return "ml_model", None

# ==================================================
# RESULT FORMATTING
# ==================================================
def format_result(prediction, risk):
    if "Malignant" in prediction:
        recommendation = "Consult a certified oncologist for further medical evaluation."
    else:
        recommendation = "Routine medical checkups are advised."
    return f"Prediction: {prediction}, Risk Level: {risk}, Recommendation: {recommendation}"

# ==================================================
# Fallback Explanation Generator
# ==================================================
def generate_fallback_explanation(prediction_text):
    """Generate a simple fallback explanation when Gemini is unavailable"""
    if "Malignant" in prediction_text:
        if "High" in prediction_text:
            return "This result indicates a high likelihood of cancer. The 'High Risk' classification means immediate medical attention is crucial. Please schedule an appointment with an oncologist as soon as possible for further diagnostic tests and treatment planning. Early intervention is important for the best possible outcome."
        elif "Moderate" in prediction_text:
            return "This result suggests the presence of cancer cells with moderate risk. While not the highest urgency, prompt medical consultation is important. An oncologist can perform additional tests to confirm the diagnosis and discuss appropriate treatment options. Don't delay follow-up care."
    else:
        if "Low" in prediction_text:
            return "This result suggests no cancer cells were detected, and the risk is low. Continue with regular annual checkups and screenings as recommended by your healthcare provider. Maintain a healthy lifestyle with regular exercise and balanced nutrition."
        else:
            return "The analysis indicates benign (non-cancerous) characteristics. Continue with routine medical checkups as advised by your doctor. Regular monitoring helps ensure ongoing breast health."
    return "Please consult with your healthcare provider to discuss these results in detail."

# ==================================================
# Gemini AI Explanation
# ==================================================
def generate_gemini_explanation(prediction_text):
    """
    Generates patient-friendly explanation using Gemini AI
    """
    if not GENAI_KEY or client is None:
        return generate_fallback_explanation(prediction_text)
    
    prompt = f"""Explain this breast cancer screening result to a patient in simple, compassionate terms:

    {prediction_text}

    Please provide:
    1. What this diagnosis means in everyday language
    2. Why this specific risk level was assigned
    3. What the recommendation involves
    4. What immediate next steps should be taken
    5. A reassuring, hopeful tone

    Keep the explanation clear, under 250 words, and suitable for someone without medical training."""
    
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",  # You can also use "gemini-1.5-pro" if available
            contents=prompt,
            config=GenerateContentConfig(
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                max_output_tokens=500,
            )
        )
        return response.text
    except Exception as e:
        app.logger.error(f"Gemini API error: {str(e)}")
        return generate_fallback_explanation(prediction_text)

# ==================================================
# WEB ROUTES
# ==================================================
@app.route("/")
def home():
    return render_template(
        "index.html",
        features_bc=selected_features_bc,
        demo_values_bc=demo_values_bc,
        prediction_text=None,
        gemini_explanation=None
    )

@app.route("/predict_bc", methods=["POST"])
def predict_bc():
    try:
        feature_vals = [
            float(request.form.get(f"feature_bc{i+1}", 0))
            for i in range(len(selected_features_bc))
        ]
        
        # Validate input values
        if any(val < 0 for val in feature_vals):
            return render_template(
                "index.html",
                features_bc=selected_features_bc,
                demo_values_bc=demo_values_bc,
                prediction_text="Error: All feature values must be positive numbers.",
                gemini_explanation="Please enter valid positive numbers for all features."
            )
        
        prediction, risk = predict_breast_manual(feature_vals)

        if prediction == "ml_model":
            df = pd.DataFrame([feature_vals], columns=selected_features_bc)
            pred = model_bc.predict(df)[0]
            prediction = "Malignant (Cancerous)" if pred == 0 else "Benign (Non-Cancerous)"
            risk = "Moderate"

        prediction_text = format_result(prediction, risk)
        gemini_explanation = generate_gemini_explanation(prediction_text)

        return render_template(
            "index.html",
            features_bc=selected_features_bc,
            demo_values_bc=demo_values_bc,
            prediction_text=prediction_text,
            gemini_explanation=gemini_explanation
        )
        
    except ValueError as e:
        return render_template(
            "index.html",
            features_bc=selected_features_bc,
            demo_values_bc=demo_values_bc,
            prediction_text="Error: Please enter valid numeric values for all features.",
            gemini_explanation="All input fields must contain numbers. Please check your entries and try again."
        )
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return render_template(
            "index.html",
            features_bc=selected_features_bc,
            demo_values_bc=demo_values_bc,
            prediction_text="An unexpected error occurred. Please try again.",
            gemini_explanation="System error. Please ensure all inputs are correctly filled and try again."
        )

# ==================================================
# RUN APP
# ==================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
