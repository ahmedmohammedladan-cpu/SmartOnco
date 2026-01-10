from flask import Flask, render_template, request, jsonify, session
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, accuracy_score, classification_report, confusion_matrix
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.secret_key = 'smartonco_secret_key_123'  # Required for sessions

# ==================================================
# BREAST CANCER MODEL
# ==================================================
selected_features_bc = [
    "worst radius", "mean concave points", "worst perimeter", "mean concavity",
    "worst concave points", "mean radius", "worst area", "mean perimeter",
    "mean texture", "worst smoothness"
]

# Load breast cancer data
data_bc = load_breast_cancer()
indices_bc = [list(data_bc.feature_names).index(f) for f in selected_features_bc]
X_bc = data_bc.data[:, indices_bc]
y_bc = data_bc.target
target_names_bc = data_bc.target_names.tolist()  # ['malignant', 'benign']

# Train-test split for breast cancer
X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(
    X_bc, y_bc, test_size=0.2, random_state=42
)

# Breast cancer model
model_bc = DecisionTreeClassifier(random_state=42)
model_bc.fit(X_train_bc, y_train_bc)
demo_values_bc = X_test_bc[0].tolist()

# ==================================================
# LUNG CANCER - Simulated Data
# ==================================================
def generate_lung_cancer_data(n_samples=1000):
    """Generate synthetic lung cancer data for demonstration"""
    np.random.seed(42)
    
    # Generate features
    age = np.random.normal(60, 15, n_samples).clip(20, 90)
    smoking = np.random.choice([1, 2], n_samples, p=[0.3, 0.7])  # 1: No, 2: Yes
    coughing = np.random.choice([1, 2], n_samples, p=[0.4, 0.6])
    chest_pain = np.random.choice([1, 2], n_samples, p=[0.5, 0.5])
    
    # Generate target based on simple rules (for simulation)
    risk_score = np.zeros(n_samples)
    risk_score += (age > 60) * 3
    risk_score += (smoking == 2) * 4
    risk_score += (coughing == 2) * 2
    risk_score += (chest_pain == 2) * 3
    
    # Create labels: 0=No Cancer, 1=Suspicious, 2=Cancer
    y_lung = np.zeros(n_samples, dtype=int)
    y_lung[risk_score >= 7] = 2  # Cancer
    y_lung[(risk_score >= 4) & (risk_score < 7)] = 1  # Suspicious
    
    # Create feature matrix
    X_lung = np.column_stack([
        np.random.randn(n_samples),  # Feature 1
        age,
        smoking,
        np.random.randn(n_samples),  # Feature 4
        np.random.randn(n_samples),  # Feature 5
        np.random.randn(n_samples),  # Feature 6
        np.random.randn(n_samples),  # Feature 7
        np.random.randn(n_samples),  # Feature 8
        np.random.randn(n_samples),  # Feature 9
        np.random.randn(n_samples),  # Feature 10
        np.random.randn(n_samples),  # Feature 11
        coughing,
        np.random.randn(n_samples),  # Feature 13
        np.random.randn(n_samples),  # Feature 14
        chest_pain
    ])
    
    return X_lung, y_lung

# Generate lung cancer data
X_lung, y_lung = generate_lung_cancer_data(1000)
target_names_lung = ['No Cancer', 'Suspicious', 'Cancer']

# Train-test split for lung cancer
X_train_lung, X_test_lung, y_train_lung, y_test_lung = train_test_split(
    X_lung, y_lung, test_size=0.2, random_state=42
)

# Lung cancer model
model_lung = DecisionTreeClassifier(random_state=42)
model_lung.fit(X_train_lung, y_train_lung)

# ==================================================
# PROSTATE CANCER - Simulated Data
# ==================================================
def generate_prostate_cancer_data(n_samples=800):
    """Generate synthetic prostate cancer data for demonstration"""
    np.random.seed(42)
    
    # Generate features
    age = np.random.normal(65, 10, n_samples).clip(40, 90)
    psa = np.random.exponential(5, n_samples).clip(0.1, 100)
    biopsy = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # 0: Negative, 1: Positive
    
    # Generate target based on PSA levels and biopsy
    y_prostate = np.zeros(n_samples, dtype=int)
    y_prostate[(psa > 10) | (biopsy == 1)] = 1  # Cancer
    
    # Add some noise
    noise = np.random.rand(n_samples) < 0.1
    y_prostate[noise] = 1 - y_prostate[noise]
    
    # Create feature matrix
    X_prostate = np.column_stack([
        age,
        psa,
        biopsy,
        np.random.randn(n_samples),  # Feature 4
        np.random.randn(n_samples),  # Feature 5
        np.random.randn(n_samples),  # Feature 6
        np.random.randn(n_samples),  # Feature 7
        np.random.randn(n_samples),  # Feature 8
        np.random.randn(n_samples),  # Feature 9
        np.random.randn(n_samples)   # Feature 10
    ])
    
    return X_prostate, y_prostate

# Generate prostate cancer data
X_prostate, y_prostate = generate_prostate_cancer_data(800)
target_names_prostate = ['No Cancer', 'Cancer']

# Train-test split for prostate cancer
X_train_prostate, X_test_prostate, y_train_prostate, y_test_prostate = train_test_split(
    X_prostate, y_prostate, test_size=0.2, random_state=42
)

# Prostate cancer model
model_prostate = DecisionTreeClassifier(random_state=42)
model_prostate.fit(X_train_prostate, y_train_prostate)

# ==================================================
# CLUSTERING FUNCTIONS
# ==================================================
def perform_clustering(data_type, n_clusters=2):
    """Perform K-means clustering and return results"""
    try:
        if data_type == "breast":
            data = X_bc
            true_labels = y_bc
            feature_names = selected_features_bc
        elif data_type == "lung":
            data = X_lung
            true_labels = y_lung
            feature_names = [f"Feature_{i+1}" for i in range(X_lung.shape[1])]
        elif data_type == "prostate":
            data = X_prostate
            true_labels = y_prostate
            feature_names = [f"Feature_{i+1}" for i in range(X_prostate.shape[1])]
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
        
        # Calculate Adjusted Rand Index if true labels are available
        ari = None
        if true_labels is not None:
            try:
                ari = adjusted_rand_score(true_labels, cluster_labels)
            except:
                ari = None
        
        # Perform PCA for visualization
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data_scaled)
        
        # Create visualization
        plt.figure(figsize=(10, 7))
        
        if true_labels is not None:
            # Plot with true labels for comparison
            plt.subplot(1, 2, 1)
            scatter1 = plt.scatter(data_pca[:, 0], data_pca[:, 1], 
                                  c=true_labels, cmap='tab10', 
                                  alpha=0.6, edgecolors='w', s=50)
            plt.colorbar(scatter1, label='True Labels')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.title(f'{data_type.capitalize()} - True Labels')
            
            plt.subplot(1, 2, 2)
            scatter2 = plt.scatter(data_pca[:, 0], data_pca[:, 1], 
                                  c=cluster_labels, cmap='tab10', 
                                  alpha=0.6, edgecolors='w', s=50)
            plt.colorbar(scatter2, label='Cluster Labels')
            plt.xlabel('PCA Component 1')
            plt.title(f'K-means Clustering (K={n_clusters})')
        else:
            # Plot only clustering results
            scatter = plt.scatter(data_pca[:, 0], data_pca[:, 1], 
                                 c=cluster_labels, cmap='tab10', 
                                 alpha=0.6, edgecolors='w', s=50)
            plt.colorbar(scatter, label='Cluster')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.title(f'{data_type.capitalize()} Cancer Data Clustering (K={n_clusters})')
        
        plt.tight_layout()
        
        # Save plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        # Get cluster statistics
        cluster_stats = []
        for i in range(n_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            cluster_data = data[cluster_indices]
            stats = {
                'cluster': i,
                'size': len(cluster_indices),
                'centroid': kmeans.cluster_centers_[i].tolist() if len(cluster_indices) > 0 else []
            }
            cluster_stats.append(stats)
        
        return {
            "success": True,
            "cluster_labels": cluster_labels.tolist(),
            "silhouette_score": float(silhouette_avg),
            "ari": float(ari) if ari is not None else None,
            "plot_url": plot_url,
            "cluster_stats": cluster_stats,
            "n_clusters": n_clusters,
            "feature_names": feature_names
        }
        
    except Exception as e:
        return {"error": str(e), "success": False}

# ==================================================
# SELF-TEST FUNCTIONS
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
            target_names = target_names_bc
        elif data_type == "lung":
            model = model_lung
            X_test = X_test_lung
            y_test = y_test_lung
            X_train = X_train_lung
            y_train = y_train_lung
            target_names = target_names_lung
        elif data_type == "prostate":
            model = model_prostate
            X_test = X_test_prostate
            y_test = y_test_prostate
            X_train = X_train_prostate
            y_train = y_train_prostate
            target_names = target_names_prostate
        else:
            return {"error": f"Unknown data type: {data_type}"}
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
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
            "cross_val_scores": cv_scores.tolist(),
            "cross_val_mean": float(cv_scores.mean()),
            "cross_val_std": float(cv_scores.std()),
            "confusion_matrix": cm.tolist(),
            "classification_report": report_dict,
            "target_names": target_names,
            "model_type": str(type(model).__name__),
            "data_type": data_type.capitalize()
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
    if "Malignant" in prediction or "Cancer Detected" in prediction:
        recommendation = "Consult a certified oncologist for further medical evaluation."
    elif "Suspicious" in prediction:
        recommendation = "Seek professional medical consultation for confirmation."
    else:
        recommendation = "Routine medical checkups are advised."

    result = f"Prediction: {prediction}, Risk Level: {risk}, Recommendation: {recommendation}"
    return result

# ==================================================
# WEB ROUTES - MAIN PAGES
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
# CLUSTERING ROUTES
# ==================================================
@app.route("/cluster/<data_type>")
def cluster_page(data_type):
    """Render clustering page for specific cancer type"""
    if data_type not in ["breast", "lung", "prostate"]:
        return "Invalid cancer type", 404
    
    # Get number of clusters from query parameter
    n_clusters = request.args.get('n_clusters', default=2, type=int)
    
    # Perform clustering
    result = perform_clustering(data_type, n_clusters)
    
    if not result.get("success"):
        return render_template("error.html", error=result.get("error", "Unknown error"))
    
    # Prepare data for template
    ari = result.get("ari", 0.0)
    if ari is None:
        ari = 0.0
    
    return render_template(
        "clustering.html",
        ari=ari,
        plot_url=result["plot_url"],
        cancer_type=data_type.capitalize(),
        n_clusters=n_clusters,
        silhouette_score=result.get("silhouette_score", 0.0)
    )

@app.route("/cluster_form/<data_type>")
def cluster_form_page(data_type):
    """Render clustering form page where user can select parameters"""
    if data_type not in ["breast", "lung", "prostate"]:
        return "Invalid cancer type", 404
    
    return render_template("cluster_form.html", cancer_type=data_type.capitalize())

# ==================================================
# SELF-TEST ROUTES
# ==================================================
@app.route("/selftest/<data_type>")
def selftest_page(data_type):
    """Render self-test page for specific cancer type"""
    if data_type not in ["breast", "lung", "prostate"]:
        return "Invalid cancer type", 404
    
    # Perform self-test
    result = perform_self_test(data_type)
    
    if not result.get("success"):
        return render_template("error.html", error=result.get("error", "Unknown error"))
    
    # Prepare data for template
    accuracy = result["accuracy"]
    cm = result["confusion_matrix"]
    report = result["classification_report"]
    target_names = result["target_names"]
    
    return render_template(
        "selftest.html",
        accuracy=accuracy,
        cm=cm,
        report=report,
        target_names=target_names,
        cancer_type=data_type.capitalize(),
        cross_val_mean=result.get("cross_val_mean", 0.0),
        cross_val_std=result.get("cross_val_std", 0.0)
    )

# ==================================================
# API ROUTES
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
            "recommendation": full_result.split(", ")[2].replace("Recommendation: ", ""),
            "full_result": full_result,
            "disclaimer": "This system is for decision support only and does not replace professional medical diagnosis."
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": "Invalid request", "details": str(e)}), 500

@app.route("/api/cluster/<data_type>", methods=["POST"])
def api_cluster_data(data_type):
    """API endpoint for clustering"""
    try:
        data = request.get_json()
        n_clusters = data.get("n_clusters", 2)
        result = perform_clustering(data_type, n_clusters)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@app.route("/api/selftest/<data_type>", methods=["GET"])
def api_self_test(data_type):
    """API endpoint for self-test"""
    result = perform_self_test(data_type)
    return jsonify(result)

# ==================================================
# NAVIGATION ROUTES
# ==================================================
@app.route("/breast/clustering")
def breast_clustering_page():
    return render_template("cluster_form.html", cancer_type="Breast")

@app.route("/breast/selftest")
def breast_selftest_page():
    result = perform_self_test("breast")
    if not result.get("success"):
        return render_template("error.html", error=result.get("error"))
    
    return render_template(
        "selftest.html",
        accuracy=result["accuracy"],
        cm=result["confusion_matrix"],
        report=result["classification_report"],
        target_names=result["target_names"],
        cancer_type="Breast"
    )

@app.route("/lung/clustering")
def lung_clustering_page():
    return render_template("cluster_form.html", cancer_type="Lung")

@app.route("/lung/selftest")
def lung_selftest_page():
    result = perform_self_test("lung")
    if not result.get("success"):
        return render_template("error.html", error=result.get("error"))
    
    return render_template(
        "selftest.html",
        accuracy=result["accuracy"],
        cm=result["confusion_matrix"],
        report=result["classification_report"],
        target_names=result["target_names"],
        cancer_type="Lung"
    )

@app.route("/prostate/clustering")
def prostate_clustering_page():
    return render_template("cluster_form.html", cancer_type="Prostate")

@app.route("/prostate/selftest")
def prostate_selftest_page():
    result = perform_self_test("prostate")
    if not result.get("success"):
        return render_template("error.html", error=result.get("error"))
    
    return render_template(
        "selftest.html",
        accuracy=result["accuracy"],
        cm=result["confusion_matrix"],
        report=result["classification_report"],
        target_names=result["target_names"],
        cancer_type="Prostate"
    )

# ==================================================
# ERROR HANDLING
# ==================================================
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', error="Internal server error"), 500

# ==================================================
# HEALTH CHECK
# ==================================================
@app.route("/health")
def health_check():
    return jsonify({
        "status": "healthy",
        "models_loaded": {
            "breast_cancer": True,
            "lung_cancer": True,
            "prostate_cancer": True
        },
        "endpoints": {
            "clustering": "/cluster/<breast|lung|prostate>",
            "self_test": "/selftest/<breast|lung|prostate>",
            "api_cluster": "/api/cluster/<type>",
            "api_selftest": "/api/selftest/<type>"
        }
    })

# ==================================================
# RUN APP
# ==================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
