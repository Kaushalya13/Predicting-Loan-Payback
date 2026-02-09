from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

app = Flask(__name__)

# Load artifacts
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / 'results/loan_payback_nn_model.pkl')
feature_columns = joblib.load(BASE_DIR / 'results/model_columns.pkl')
scaler = joblib.load(BASE_DIR / 'results/scaler.pkl')
encoders = joblib.load(BASE_DIR / 'results/encoders.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # 1. Get raw data from form
        data = {
            "annual_income": float(request.form.get("annual_income")),
            "debt_to_income_ratio": float(request.form.get("debt_to_income_ratio")),
            "credit_score": int(request.form.get("credit_score")),
            "loan_amount": float(request.form.get("loan_amount")),
            "interest_rate": float(request.form.get("interest_rate")),
            "gender": request.form.get("gender"),
            "marital_status": request.form.get("marital_status"),
            "education_level": request.form.get("education_level"),
            "employment_status": request.form.get("employment_status"),
            "loan_purpose": request.form.get("loan_purpose"),
            "grade_subgrade": request.form.get("grade_subgrade")
        }

        # 2. Create DataFrame
        X_input = pd.DataFrame([data])

        # 3. Feature Engineering (Crucial fix for your KeyError)
        X_input['loan_to_income_ratio'] = X_input['loan_amount'] / (X_input['annual_income'] + 1)
        X_input['monthly_debt_burden'] = (X_input['annual_income'] * X_input['debt_to_income_ratio']) / 12
        X_input['expected_interest_cost'] = X_input['loan_amount'] * (X_input['interest_rate'] / 100)
        X_input['credit_per_interest'] = X_input['credit_score'] / (X_input['interest_rate'] + 1)

        # 4. Apply Label Encoders
        for col, le in encoders.items():
            X_input[col] = le.transform(X_input[col].astype(str))

        # 5. Reorder columns to match exactly what the model saw during training
        X_input = X_input[feature_columns]

        # 6. Scale and Predict
        X_scaled = scaler.transform(X_input)
        prob = float(model.predict_proba(X_scaled)[0][1])

        # Result mapping
        status = "Low Risk" if prob >= 0.7 else ("Medium Risk" if prob >= 0.4 else "High Risk")
        color = "#28a745" if prob >= 0.7 else ("#ffc107" if prob >= 0.4 else "#dc3545")

        return render_template('index.html', 
                               prediction_text=f"Payback Probability: {prob:.2%}",
                               status=f"Assessment: {status}",
                               color=color)

if __name__ == "__main__":
    app.run(debug=True)