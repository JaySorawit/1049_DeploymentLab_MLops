import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save to app folder
with open("app/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Classification model saved to app/model.pkl")