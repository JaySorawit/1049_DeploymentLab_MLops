import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the housing dataset (make sure the file path is correct)
df = pd.read_csv("Housing.csv")

# Convert categorical features into numerical using one-hot encoding
df_encoded = pd.get_dummies(df)

# Separate features (X) and target (y)
X = df_encoded.drop("price", axis=1)  # Features: all columns except 'price'
y = df_encoded["price"]               # Target: the column we want to predict

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model to a file using pickle
with open("app/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save the feature columns used during training for later use
with open("app/feature_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("Model and feature columns saved to app/")
