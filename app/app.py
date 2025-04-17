from flask import Flask, request, jsonify
import traceback
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained ML model (used in all exercises)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the feature columns used during training
# Uncomment the following lines if you want to use feature columns for validation
# with open("feature_columns.pkl", "rb") as f:
#     feature_columns = pickle.load(f)

@app.route("/")
def home():
    return "ML Model is Running"

# Exercise 1, 2, 3: Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Exercise 3: Add Input Validation
    if "features" not in data:
        return jsonify({"error": "'features' key is missing"}), 400

    try:
        # Exercise 3: Add Input Validation
        X = np.array(data["features"], dtype=float)
    except ValueError:
        return jsonify({"error": "All inputs must be numeric"}), 400

    # Exercise 2: Handle both single and multiple inputs
    if len(X.shape) == 1:
        X = X.reshape(1, -1)

    # Exercise 3: Add Input Validation
    if X.shape[1] != 4:
        return jsonify({"error": "Each input must have exactly 4 features"}), 400

    # Make predictions
    predictions = model.predict(X).tolist()

    # Exercise 1: Return confidence scores if model supports predict_proba (for classifiers)
    if hasattr(model, "predict_proba"):
        confidences = model.predict_proba(X).max(axis=1).tolist()
        return jsonify({
            "predictions": predictions,
            "confidences": [round(c, 2) for c in confidences]
        })

    # Return prediction only (for regression models)
    return jsonify({"predictions": predictions})

# Exercise 5: Train a new model on a provided housing dataset. The task is to build a regression model to predict the housing price.
# Uncomment the following lines to implement to predict the housing price
# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.get_json()

#     if "features" not in data:
#         return jsonify({"error": "'features' key is missing"}), 400

#     try:
#         # Convert input data to a NumPy array (ensure it's numeric)
#         X = np.array(data["features"], dtype=float)
#     except ValueError:
#         return jsonify({"error": "All inputs must be numeric"}), 400

#     if len(X.shape) == 1:
#         X = X.reshape(1, -1)  # Ensure input is a 2D array

#     # Check if the input has the correct number of features (match with the model)
#     if len(X[0]) != len(feature_columns):
#         return jsonify({"error": f"Input must have exactly {len(feature_columns)} features"}), 400

#     # Make predictions using the model
#     predictions = model.predict(X).tolist()

#     return jsonify({"predictions": predictions})

# Exercise 4: Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)
