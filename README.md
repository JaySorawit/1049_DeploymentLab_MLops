# Sorawit Mokthaisong 64070501049

### CPE393 SPECIAL TOPIC III : MACHINE LEARNING OPERATION [Lab-Model Deployment]

This lab, I learned how to deploy a machine learning model using Flask and Docker. It includes both a **classification model** (Iris dataset) and a **regression model** (Housing dataset), and implements API features like input validation, confidence scores, health checks, and multi-input support.

---

## 📌 Features

- `/predict` endpoint:
  - Supports multiple inputs
  - Returns classification or regression results
  - Includes confidence scores (for classifiers)
  - Validates input format
- `/health` endpoint to check if API is live
- Dockerized for consistent deployment

---

## ⚙️ Setup & Usage

### 1. Clone the repo
```bash
git clone https://github.com/JaySorawit/1049_DeploymentLab_MLops.git
cd 1049_DeploymentLab_MLops.git
```
### 2. Train model
Run either of the following:

For classification (Iris)
```bash
python save_model.py
```
For regression (Housing)
Make sure Housing.csv is in the root folder.

```bash
python save_housing_model.py
```
This will save model.pkl into the app/ folder.

### 3. Build Docker image
```bash
docker build -t ml-api .
```
### 4. Run the container
```bash
docker run -p 9000:9000 ml-api
```

## 🚀 API Endpoints
### GET /health
Returns:

```json
{
  "status": "ok"
}
```

### POST /predict
🔸 Request (Classification)
```json
{
  "features": [
    [5.1, 3.5, 1.4, 0.2],
    [6.2, 3.4, 5.4, 2.3]
  ]
}
```
🔹 Response (Classification)
```json
{
  "predictions": [0, 2],
  "confidences": [0.97, 0.89]
}
```
🔸 Request (Regression)
```json
{
  "features": [
    [7420, 4, 2, 3, 1, 0, 0, 0, 1, 1, 2, 1, 0, 1, 0, 0, 1, 0, 0, 0],
    [8960, 4, 4, 4, 1, 0, 0, 0, 1, 1, 3, 0, 1, 0, 0, 0, 0, 1, 0, 1]
  ]
}
```
🔹 Response (Regression)
```json
{
  "predictions": [246800.55, 152000.23]
}
```
🔸 Invalid input example
```json
{
  "features": [5.1, 3.5]
}
```
🔹 Returns:
```json
{
  "error": "Each input must have exactly 4 features"
}
```
