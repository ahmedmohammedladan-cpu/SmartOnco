from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, adjusted_rand_score, silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import os

# =========================
# Gemini AI Import (for google-genai>=1.1.0)
# =========================
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Google AI SDK not available. Using fallback explanations.")

# =========================
# Initialize Flask
# =========================
app = Flask(__name__)

# =========================
# Gemini AI Configuration
# =========================
GENAI_API_KEY = os.environ.get("GENAI_API_KEY", "")

if GEMINI_AVAILABLE and GENAI_API_KEY:
    try:
        client = genai.Client(api_key=GENAI_API_KEY)
        GEMINI_ENABLED = True
        print("Gemini AI initialized successfully")
    except Exception as e:
        print(f"Failed to initialize Gemini: {e}")
        GEMINI_ENABLED = False
        client = None
else:
    GEMINI_ENABLED = False
    client = None

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
# SIMULATED DATA FOR LUNG AND PROSTATE (for clustering/self-test)
# ==================================================
# Create simple simulated data for lung cancer (15 features as per your form)
np.random.seed(42)
n_samples = 300

# Lung cancer simulation
X_lung_sim = np.random.randn(n_samples, 15)
# Add some structure
X_lung_sim[:, 1] += np.random.randn(n_samples) * 0.5  # Age-like feature
X_lung_sim[:, 2] += np.random.randn(n_samples) * 0.5  # Smoking-like feature
X_lung_sim[:, 11] += np.random.randn(n_samples) * 0.5  # Coughing-like feature
X_lung_sim[:, 14] += np.random.randn(n_samples) * 0.5  # Chest pain-like feature

y_lung_sim = np.zeros(n_samples, dtype=int)
score = X_lung_sim[:, 1] + X_lung_sim[:, 2] * 1.5 + X_lung_sim[:, 11] + X_lung_sim[:, 14]
y_lung_sim[score > 1] = 1
y_lung_sim[score > 2.5] = 2

# Prostate cancer simulation
X_prostate_sim = np.random.randn(n_samples, 10)
X_prostate_sim[:, 1] += np.random.randn(n_samples) * 0.5  # PSA-like feature
X_prostate_sim[:, 2] += np.random.randn(n_samples) * 0.5  # Biopsy-like feature
y_prostate_sim = ((X_prostate_sim[:, 1] > 0.5) | (X_prostate_sim[:, 2] > 0.7)).astype(int)

# Initialize clustering for lung and prostate
scaler_lung = StandardScaler()
X_scaled_lung = scaler_lung.fit_transform(X_lung_sim)
kmeans_lung = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_lung.fit(X_scaled_lung)

scaler_prostate = StandardScaler()
X_scaled_prostate = scaler_prostate.fit_transform(X_prostate_sim)
kmeans_prostate = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_prostate.fit(X_scaled_prostate)

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
# PROSTATE CANCER – MANUAL RULE SYSTEM
# ==================================================
def predict_prostate_manual(features):
    psa = float(features[1])
    biopsy = int(features[2])

    if biopsy == 1 or psa > 10:
        return "Prostate Cancer Detected", "High"
    else:
        return "No Prostate Cancer Detected", "Low"

# ==================================================
# RESULT FORMATTING
# ==================================================
def format_result(prediction, risk):
    """
    Returns a professionally formatted result separated by comma and space
    """
    if "Malignant" in prediction or "Cancer Detected" in prediction:
        recommendation = "Consult a certified oncologist for further medical evaluation."
    elif "Suspicious" in prediction:
        recommendation = "Seek professional medical consultation for confirmation."
    else:
        recommendation = "Routine medical checkups are advised."

    # Join all info in one line
    result = f"Prediction: {prediction}, Risk Level: {risk}, Recommendation: {recommendation}"
    return result

# ==================================================
# GEMINI AI EXPLANATIONS FOR ALL CANCER TYPES
# ==================================================
def generate_fallback_explanation(prediction_text, cancer_type="breast"):
    """Generate explanation when Gemini is unavailable"""
    cancer_type = cancer_type.lower()
    
    if "Malignant" in prediction_text or "Cancer Detected" in prediction_text:
        if "High" in prediction_text:
            return f"🚨 **High Risk Alert**: This indicates a strong likelihood of {cancer_type} cancer. Immediate consultation with an oncologist is crucial for further testing and treatment planning."
        elif "Moderate" in prediction_text:
            return f"⚠️ **Moderate Risk**: This suggests possible {cancer_type} cancer. Schedule an appointment with an oncologist for further evaluation and appropriate next steps."
        else:
            return f"🔍 **Medical Consultation Needed**: This {cancer_type} cancer result requires further evaluation by a healthcare professional."
    elif "Suspicious" in prediction_text:
        return f"⚠️ **Suspicious Findings**: Further investigation is recommended for potential {cancer_type} cancer. Please consult with a healthcare provider."
    else:
        return f"✅ **Normal/Benign Result**: No signs of {cancer_type} cancer detected. Continue with routine medical checkups as advised by your doctor."

def generate_gemini_explanation(prediction_text, cancer_type="breast"):
    """
    Generates patient-friendly explanation using Gemini AI for all cancer types
    """
    if not GEMINI_ENABLED or client is None:
        return generate_fallback_explanation(prediction_text, cancer_type)
    
    prompt = f"""Explain this {cancer_type} cancer screening result in simple, compassionate terms:

{prediction_text}

Please provide a clear explanation in under 150 words. Use simple language, be reassuring, and include:
1. What the result means in plain language
2. What the risk level indicates
3. Recommended next steps
4. Words of encouragement and support"""

    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config={
                "temperature": 0.7,
                "top_p": 0.8,
                "max_output_tokens": 300,
            }
        )
        return response.text
        
    except Exception as e:
        error_msg = str(e)
        print(f"Gemini API Error: {error_msg}")
        
        if "PERMISSION_DENIED" in error_msg or "403" in error_msg:
            return "⚠️ **API Access Issue**: Please ensure Generative Language API is enabled in Google Cloud Console. " + generate_fallback_explanation(prediction_text, cancer_type)
        elif "quota" in error_msg.lower():
            return "📊 **API Limit Reached**: " + generate_fallback_explanation(prediction_text, cancer_type)
        else:
            return generate_fallback_explanation(prediction_text, cancer_type)

# ==================================================
# CLUSTERING FUNCTIONS
# ==================================================
def create_clustering_plot(data_scaled, labels, true_labels, title):
    """Create clustering visualization plot"""
    plt.figure(figsize=(12, 5))
    
    # True labels plot
    plt.subplot(1, 2, 1)
    scatter1 = plt.scatter(data_scaled[:, 0], data_scaled[:, 1], 
                          c=true_labels, cmap='viridis', 
                          alpha=0.6, edgecolors='w', s=50)
    plt.colorbar(scatter1, label='True Labels')
    plt.xlabel('Standardized Feature 1')
    plt.ylabel('Standardized Feature 2')
    plt.title(f'{title} - True Labels')
    
    # Cluster labels plot
    plt.subplot(1, 2, 2)
    scatter2 = plt.scatter(data_scaled[:, 0], data_scaled[:, 1], 
                          c=labels, cmap='viridis', 
                          alpha=0.6, edgecolors='w', s=50)
    plt.colorbar(scatter2, label='Cluster Labels')
    plt.xlabel('Standardized Feature 1')
    plt.ylabel('Standardized Feature 2')
    plt.title(f'{title} - K-means Clustering')
    
    plt.tight_layout()
    
    # Save plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

# ==================================================
# WEB ROUTES
# ==================================================
@app.route("/")
def home():
    return render_template(
        "index.html",
        features_bc=selected_features_bc,
        demo_values_bc=demo_values_bc
    )

@app.route("/predict_bc", methods=["POST"])
def predict_bc():
    try:
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

        prediction_text = format_result(prediction, risk)
        gemini_explanation = generate_gemini_explanation(prediction_text, "breast")

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
            gemini_explanation="All input fields must contain numbers. Please check your entries."
        )
    except Exception as e:
        print(f"Prediction error: {e}")
        return render_template(
            "index.html",
            features_bc=selected_features_bc,
            demo_values_bc=demo_values_bc,
            prediction_text="An unexpected error occurred.",
            gemini_explanation="Please try again with valid inputs."
        )

@app.route("/lung")
def lung_page():
    return render_template("lung_cancer.html")

@app.route("/predict_lc", methods=["POST"])
def predict_lc():
    try:
        feature_vals = [
            float(request.form.get(f"feature_lc{i+1}", 1))
            for i in range(15)
        ]

        prediction, risk = predict_lung_manual(feature_vals)
        prediction_text = format_result(prediction, risk)
        gemini_explanation = generate_gemini_explanation(prediction_text, "lung")

        return render_template(
            "lung_cancer.html",
            prediction_text=prediction_text,
            gemini_explanation=gemini_explanation
        )
        
    except Exception as e:
        print(f"Lung prediction error: {e}")
        return render_template(
            "lung_cancer.html",
            prediction_text="An error occurred. Please check your inputs.",
            gemini_explanation="Please try again with valid inputs."
        )

@app.route("/prostate")
def prostate_page():
    return render_template("prostate.html")

@app.route("/predict_prostate", methods=["POST"])
def predict_prostate():
    try:
        psa = float(request.form.get("feature_prostate2", 0))
        biopsy = int(request.form.get("feature_prostate3", 0))
        
        feature_vals = [0] * 10  # Placeholder for other features
        feature_vals[1] = psa
        feature_vals[2] = biopsy

        prediction, risk = predict_prostate_manual(feature_vals)
        prediction_text = format_result(prediction, risk)
        gemini_explanation = generate_gemini_explanation(prediction_text, "prostate")

        return render_template(
            "prostate.html",
            prediction_text=prediction_text,
            gemini_explanation=gemini_explanation
        )
        
    except Exception as e:
        print(f"Prostate prediction error: {e}")
        return render_template(
            "prostate.html",
            prediction_text="An error occurred. Please check your inputs.",
            gemini_explanation="Please try again with valid inputs."
        )

# ==================================================
# NEW ROUTES FOR SELF-TEST
# ==================================================
@app.route("/selftest_bc")
def selftest_bc():
    """Self-test for Breast Cancer Model"""
    y_pred = model_bc.predict(X_test_bc)
    acc = accuracy_score(y_test_bc, y_pred)
    cm = confusion_matrix(y_test_bc, y_pred).tolist()
    report = classification_report(y_test_bc, y_pred, target_names=data_bc.target_names, output_dict=True)
    
    # Convert report to match your template structure
    formatted_report = {}
    for cls in data_bc.target_names:
        formatted_report[cls] = {
            'precision': report[cls]['precision'],
            'recall': report[cls]['recall'],
            'f1-score': report[cls]['f1-score'],
            'support': int(report[cls]['support'])
        }
    
    return render_template("selftest.html", 
                         accuracy=acc, 
                         cm=cm, 
                         target_names=data_bc.target_names, 
                         report=formatted_report,
                         cancer_type="Breast Cancer")

@app.route("/selftest_lc")
def selftest_lc():
    """Self-test for Lung Cancer Model"""
    # For lung cancer, we'll use simulated data and a simple model
    model_lc = DecisionTreeClassifier(random_state=42)
    X_train_lc, X_test_lc, y_train_lc, y_test_lc = train_test_split(
        X_lung_sim, y_lung_sim, test_size=0.2, random_state=42
    )
    model_lc.fit(X_train_lc, y_train_lc)
    
    y_pred = model_lc.predict(X_test_lc)
    acc = accuracy_score(y_test_lc, y_pred)
    cm = confusion_matrix(y_test_lc, y_pred).tolist()
    report = classification_report(y_test_lc, y_pred, target_names=["No Cancer", "Suspicious", "Cancer"], output_dict=True)
    
    # Convert report to match your template structure
    formatted_report = {}
    for cls in ["No Cancer", "Suspicious", "Cancer"]:
        if cls in report:
            formatted_report[cls] = {
                'precision': report[cls]['precision'],
                'recall': report[cls]['recall'],
                'f1-score': report[cls]['f1-score'],
                'support': int(report[cls]['support'])
            }
    
    return render_template("selftest.html", 
                         accuracy=acc, 
                         cm=cm, 
                         target_names=["No Cancer", "Suspicious", "Cancer"], 
                         report=formatted_report,
                         cancer_type="Lung Cancer")

@app.route("/selftest_pc")
def selftest_pc():
    """Self-test for Prostate Cancer"""
    # For prostate cancer, we'll use simulated data and a simple model
    model_pc = DecisionTreeClassifier(random_state=42)
    X_train_pc, X_test_pc, y_train_pc, y_test_pc = train_test_split(
        X_prostate_sim, y_prostate_sim, test_size=0.2, random_state=42
    )
    model_pc.fit(X_train_pc, y_train_pc)
    
    y_pred = model_pc.predict(X_test_pc)
    acc = accuracy_score(y_test_pc, y_pred)
    cm = confusion_matrix(y_test_pc, y_pred).tolist()
    report = classification_report(y_test_pc, y_pred, target_names=["No Cancer", "Cancer"], output_dict=True)
    
    # Convert report to match your template structure
    formatted_report = {}
    for cls in ["No Cancer", "Cancer"]:
        formatted_report[cls] = {
            'precision': report[cls]['precision'],
            'recall': report[cls]['recall'],
            'f1-score': report[cls]['f1-score'],
            'support': int(report[cls]['support'])
        }
    
    return render_template("selftest.html", 
                         accuracy=acc, 
                         cm=cm, 
                         target_names=["No Cancer", "Cancer"], 
                         report=formatted_report,
                         cancer_type="Prostate Cancer")

# ==================================================
# NEW ROUTES FOR CLUSTERING
# ==================================================
@app.route("/clustering_bc")
def clustering_bc():
    """Clustering visualization for Breast Cancer"""
    labels = kmeans_bc.predict(X_scaled_bc)
    ari = adjusted_rand_score(y_bc, labels)
    silhouette = silhouette_score(X_scaled_bc, labels)
    
    # Create visualization
    plot_url = create_clustering_plot(X_scaled_bc, labels, y_bc, "Breast Cancer")
    
    return render_template(
        "clustering.html",
        ari=ari,
        plot_url=plot_url,
        cancer_type="Breast",
        n_clusters=2,
        silhouette_score=silhouette
    )

@app.route("/clustering_lc")
def clustering_lc():
    """Clustering visualization for Lung Cancer"""
    labels = kmeans_lung.predict(X_scaled_lung)
    ari = adjusted_rand_score(y_lung_sim, labels)
    silhouette = silhouette_score(X_scaled_lung, labels)
    
    # Create visualization
    plot_url = create_clustering_plot(X_scaled_lung, labels, y_lung_sim, "Lung Cancer")
    
    return render_template(
        "clustering.html",
        ari=ari,
        plot_url=plot_url,
        cancer_type="Lung",
        n_clusters=3,
        silhouette_score=silhouette
    )

@app.route("/clustering_pc")
def clustering_pc():
    """Clustering visualization for Prostate Cancer"""
    labels = kmeans_prostate.predict(X_scaled_prostate)
    ari = adjusted_rand_score(y_prostate_sim, labels)
    silhouette = silhouette_score(X_scaled_prostate, labels)
    
    # Create visualization
    plot_url = create_clustering_plot(X_scaled_prostate, labels, y_prostate_sim, "Prostate Cancer")
    
    return render_template(
        "clustering.html",
        ari=ari,
        plot_url=plot_url,
        cancer_type="Prostate",
        n_clusters=2,
        silhouette_score=silhouette
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

        prediction_text = format_result(prediction, risk)
        gemini_explanation = generate_gemini_explanation(prediction_text, "breast")

        response = {
            "prediction": prediction,
            "risk_level": risk,
            "recommendation": prediction_text.split(", ")[2].replace("Recommendation: ", ""),
            "ai_explanation": gemini_explanation,
            "full_result": prediction_text,
            "disclaimer": "This system is for decision support only and does not replace professional medical diagnosis."
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": "Invalid request", "details": str(e)}), 500

# ==================================================
# HEALTH CHECK
# ==================================================
@app.route("/health")
def health_check():
    return jsonify({
        "status": "healthy",
        "models": ["breast", "lung", "prostate"],
        "features": {
            "gemini_ai": GEMINI_ENABLED,
            "self_test": True,
            "clustering": True,
            "api": True
        },
        "endpoints": {
            "self_test": "/selftest_bc, /selftest_lc, /selftest_pc",
            "clustering": "/clustering_bc, /clustering_lc, /clustering_pc",
            "api": "/api/predict/breast"
        }
    })

# ==================================================
# RUN APP
# ==================================================
if __name__ == "__main__":
    print("🚀 SmartOnco System Started")
    print("✅ Breast Cancer: Model loaded")
    print("✅ Lung Cancer: Simulated data ready")
    print("✅ Prostate Cancer: Simulated data ready")
    print(f"✅ Gemini AI: {'Enabled' if GEMINI_ENABLED else 'Disabled'}")
    print("✅ Clustering: Available at /clustering_bc, /clustering_lc, /clustering_pc")
    print("✅ Self-Test: Available at /selftest_bc, /selftest_lc, /selftest_pc")
    print("✅ API: Available at /api/predict/breast")
    print("✅ Health Check: Available at /health")
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
