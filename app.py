from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, accuracy_score, classification_report, confusion_matrix
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
# SIMULATED DATA FOR LUNG AND PROSTATE (for clustering/self-test)
# ==================================================
# Create simple simulated data for lung cancer (15 features as per your form)
np.random.seed(42)
n_samples = 300
X_lung_sim = np.random.randn(n_samples, 15)
# Create some structure for clustering
X_lung_sim[:, 0] += np.random.randn(n_samples) * 0.5  # Age-like feature
X_lung_sim[:, 1] += np.random.randn(n_samples) * 0.5  # Smoking-like feature
X_lung_sim[:, 11] += np.random.randn(n_samples) * 0.5  # Coughing-like feature
X_lung_sim[:, 14] += np.random.randn(n_samples) * 0.5  # Chest pain-like feature

# Create labels for lung simulation (3 classes)
y_lung_sim = np.zeros(n_samples, dtype=int)
score = X_lung_sim[:, 0] + X_lung_sim[:, 1] * 1.5 + X_lung_sim[:, 11] + X_lung_sim[:, 14]
y_lung_sim[score > 1] = 1
y_lung_sim[score > 2.5] = 2

# Create simple simulated data for prostate cancer (10 features as per your form)
X_prostate_sim = np.random.randn(n_samples, 10)
# Create some structure
X_prostate_sim[:, 1] += np.random.randn(n_samples) * 0.5  # PSA-like feature
X_prostate_sim[:, 2] += np.random.randn(n_samples) * 0.5  # Biopsy-like feature

# Create labels for prostate simulation (2 classes)
y_prostate_sim = ((X_prostate_sim[:, 1] > 0.5) | (X_prostate_sim[:, 2] > 0.7)).astype(int)

# Train simple models for lung and prostate for self-test
model_lung_sim = DecisionTreeClassifier(random_state=42)
model_lung_sim.fit(X_lung_sim, y_lung_sim)

model_prostate_sim = DecisionTreeClassifier(random_state=42)
model_prostate_sim.fit(X_prostate_sim, y_prostate_sim)

# ==================================================
# CLUSTERING FUNCTION
# ==================================================
def perform_clustering(data_type, n_clusters=2):
    """Perform K-means clustering and return results"""
    try:
        if data_type == "breast":
            data = X_bc
            true_labels = y_bc
        elif data_type == "lung":
            data = X_lung_sim
            true_labels = y_lung_sim
        elif data_type == "prostate":
            data = X_prostate_sim
            true_labels = y_prostate_sim
        else:
            return {"error": f"Unknown data type: {data_type}"}
        
        # Standardize the data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data_scaled)
        
        # Calculate metrics
        silhouette_avg = silhouette_score(data_scaled, cluster_labels)
        
        # Calculate Adjusted Rand Index
        ari = adjusted_rand_score(true_labels, cluster_labels)
        
        # Perform PCA for visualization (2D)
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data_scaled)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        # Plot with true labels
        plt.subplot(2, 1, 1)
        scatter_true = plt.scatter(data_pca[:, 0], data_pca[:, 1], 
                                   c=true_labels, cmap='tab10', 
                                   alpha=0.7, edgecolors='w', s=60)
        plt.colorbar(scatter_true, label='True Labels')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'{data_type.capitalize()} Cancer - True Class Distribution')
        plt.grid(True, alpha=0.3)
        
        # Plot with cluster labels
        plt.subplot(2, 1, 2)
        scatter_cluster = plt.scatter(data_pca[:, 0], data_pca[:, 1], 
                                      c=cluster_labels, cmap='tab10', 
                                      alpha=0.7, edgecolors='w', s=60)
        plt.colorbar(scatter_cluster, label='Cluster Labels')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'K-means Clustering Results (K={n_clusters})')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return {
            "success": True,
            "ari": float(ari),
            "plot_url": plot_url,
            "silhouette_score": float(silhouette_avg),
            "n_clusters": n_clusters,
            "data_type": data_type
        }
        
    except Exception as e:
        return {"error": str(e), "success": False}

# ==================================================
# SELF-TEST FUNCTION
# ==================================================
def perform_self_test(data_type):
    """Perform comprehensive model self-testing"""
    try:
        if data_type == "breast":
            model = model_bc
            X_test = X_test_bc
            y_test = y_test_bc
            X_train = X_train_bc
            y_train = y_train_bc
            target_names = ['Malignant', 'Benign']
        elif data_type == "lung":
            model = model_lung_sim
            # Split the simulated data
            X_train, X_test, y_train, y_test = train_test_split(
                X_lung_sim, y_lung_sim, test_size=0.2, random_state=42
            )
            model.fit(X_train, y_train)  # Re-fit on train split
            target_names = ['No Cancer', 'Suspicious', 'Cancer']
        elif data_type == "prostate":
            model = model_prostate_sim
            # Split the simulated data
            X_train, X_test, y_train, y_test = train_test_split(
                X_prostate_sim, y_prostate_sim, test_size=0.2, random_state=42
            )
            model.fit(X_train, y_train)  # Re-fit on train split
            target_names = ['No Cancer', 'Cancer']
        else:
            return {"error": f"Unknown data type: {data_type}"}
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        
        return {
            "success": True,
            "accuracy": float(accuracy),
            "cross_val_mean": float(cv_scores.mean()),
            "cross_val_std": float(cv_scores.std()),
            "confusion_matrix": cm.tolist(),
            "classification_report": report_dict,
            "target_names": target_names,
            "data_type": data_type
        }
        
    except Exception as e:
        return {"error": str(e), "success": False}

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

    return render_template(
        "index.html",
        features_bc=selected_features_bc,
        demo_values_bc=demo_values_bc,
        prediction_text=prediction_text
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
    prediction_text = format_result(prediction, risk)

    return render_template(
        "lung_cancer.html",
        prediction_text=prediction_text
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

    prediction_text = format_result(prediction, risk)

    return render_template(
        "prostate.html",
        prediction_text=prediction_text
    )

# ==================================================
# NEW ROUTES FOR CLUSTERING
# ==================================================
@app.route("/breast/clustering")
def breast_clustering_page():
    """Show clustering options for breast cancer"""
    return render_template("cluster_form.html", cancer_type="breast")

@app.route("/lung/clustering")
def lung_clustering_page():
    """Show clustering options for lung cancer"""
    return render_template("cluster_form.html", cancer_type="lung")

@app.route("/prostate/clustering")
def prostate_clustering_page():
    """Show clustering options for prostate cancer"""
    return render_template("cluster_form.html", cancer_type="prostate")

@app.route("/cluster/<data_type>", methods=["GET", "POST"])
def perform_clustering_route(data_type):
    """Perform clustering and show results"""
    if data_type not in ["breast", "lung", "prostate"]:
        return "Invalid cancer type", 404
    
    if request.method == "POST":
        n_clusters = int(request.form.get("n_clusters", 2))
    else:
        n_clusters = int(request.args.get("n_clusters", 2))
    
    result = perform_clustering(data_type, n_clusters)
    
    if not result.get("success", False):
        error_msg = result.get("error", "Unknown error occurred")
        return render_template("error.html", error=error_msg)
    
    return render_template(
        "clustering.html",
        ari=result["ari"],
        plot_url=result["plot_url"],
        cancer_type=data_type.capitalize(),
        n_clusters=n_clusters,
        silhouette_score=result.get("silhouette_score", 0.0)
    )

# ==================================================
# NEW ROUTES FOR SELF-TEST
# ==================================================
@app.route("/breast/selftest")
def breast_selftest_page():
    """Run self-test for breast cancer model"""
    result = perform_self_test("breast")
    
    if not result.get("success", False):
        error_msg = result.get("error", "Unknown error occurred")
        return render_template("error.html", error=error_msg)
    
    return render_template(
        "selftest.html",
        accuracy=result["accuracy"],
        cm=result["confusion_matrix"],
        report=result["classification_report"],
        target_names=result["target_names"],
        cancer_type="Breast",
        cross_val_mean=result.get("cross_val_mean", 0),
        cross_val_std=result.get("cross_val_std", 0)
    )

@app.route("/lung/selftest")
def lung_selftest_page():
    """Run self-test for lung cancer model"""
    result = perform_self_test("lung")
    
    if not result.get("success", False):
        error_msg = result.get("error", "Unknown error occurred")
        return render_template("error.html", error=error_msg)
    
    return render_template(
        "selftest.html",
        accuracy=result["accuracy"],
        cm=result["confusion_matrix"],
        report=result["classification_report"],
        target_names=result["target_names"],
        cancer_type="Lung",
        cross_val_mean=result.get("cross_val_mean", 0),
        cross_val_std=result.get("cross_val_std", 0)
    )

@app.route("/prostate/selftest")
def prostate_selftest_page():
    """Run self-test for prostate cancer model"""
    result = perform_self_test("prostate")
    
    if not result.get("success", False):
        error_msg = result.get("error", "Unknown error occurred")
        return render_template("error.html", error=error_msg)
    
    return render_template(
        "selftest.html",
        accuracy=result["accuracy"],
        cm=result["confusion_matrix"],
        report=result["classification_report"],
        target_names=result["target_names"],
        cancer_type="Prostate",
        cross_val_mean=result.get("cross_val_mean", 0),
        cross_val_std=result.get("cross_val_std", 0)
    )

# ==================================================
# SIMPLE SELF-TEST ROUTE (for direct access)
# ==================================================
@app.route("/selftest/<data_type>")
def direct_selftest(data_type):
    """Direct self-test route"""
    if data_type == "breast":
        return breast_selftest_page()
    elif data_type == "lung":
        return lung_selftest_page()
    elif data_type == "prostate":
        return prostate_selftest_page()
    else:
        return render_template("error.html", error="Invalid cancer type")

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
            "recommendation": full_result.split(", ")[2],  # only recommendation
            "full_result": full_result,
            "disclaimer": "This system is for decision support only and does not replace professional medical diagnosis."
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": "Invalid request", "details": str(e)}), 500

# ==================================================
# ADDITIONAL API ROUTES FOR CLUSTERING AND SELF-TEST
# ==================================================
@app.route("/api/cluster/<data_type>", methods=["POST"])
def api_cluster(data_type):
    """API endpoint for clustering"""
    try:
        data = request.get_json()
        n_clusters = data.get("n_clusters", 2)
        result = perform_clustering(data_type, n_clusters)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/selftest/<data_type>", methods=["GET"])
def api_selftest(data_type):
    """API endpoint for self-test"""
    result = perform_self_test(data_type)
    return jsonify(result)

# ==================================================
# ERROR PAGE
# ==================================================
@app.route("/error")
def error_page():
    error_msg = request.args.get("msg", "An error occurred")
    return render_template("error.html", error=error_msg)

@app.errorhandler(404)
def page_not_found(e):
    return render_template("error.html", error="Page not found"), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template("error.html", error="Internal server error"), 500

# ==================================================
# HEALTH CHECK
# ==================================================
@app.route("/health")
def health_check():
    return jsonify({
        "status": "healthy",
        "models": ["breast", "lung", "prostate"],
        "endpoints": {
            "clustering": "/breast/clustering, /lung/clustering, /prostate/clustering",
            "self_test": "/breast/selftest, /lung/selftest, /prostate/selftest"
        }
    })

# ==================================================
# RUN APP
# ==================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
