Loan Payback Predictor

Loan Payback Predictor is a deep learning application for predicting loan repayment outcomes using neural networks. It processes borrower data (e.g., income, credit score, debt ratio) to forecast payback probability, built for the Kaggle Playground Series S5E11 competition.

GitHub Repository: https://github.com/Kaushalya13/Predicting-Loan-Payback

Kaggle Competition: https://www.kaggle.com/competitions/playground-series-s5e11

✨ Features

Binary classification: Predicts loan payback (1=paid back, 0=default) with probability and risk tiers (Low/Medium/High).
Feature engineering: Custom ratios like loan_to_income_ratio and monthly_debt_burden.
Real-time web app: Interactive form for new predictions via Flask.
EDA integration: Visualizations for data insights (distributions, correlations, outliers).
Ethical focus: Handles biases in features like gender and employment status.

🚀 How It Works

Data Input: Enter borrower details (annual income, credit score, etc.) via web form.
Preprocessing: Encode categoricals, scale numerics, add engineered features.
Inference: Neural network predicts probability; maps to risk level with color feedback.
Output: Displays payback chance (e.g., 78%) and assessment (e.g. Low Risk).

🛠️ Tech Stack

ML Framework: TensorFlow/Keras (Neural Network)
Data Processing: pandas, numpy, scikit-learn (encoding, scaling)
Web App: Flask (back-end API), HTML/CSS (front-end form)
Visualization: matplotlib, seaborn
Model Saving: joblib

🧠 Research & Data Assets

* This project is for CIS6005 Computational Intelligence module, focusing on deep learning for financial prediction.

* Algorithm: Multi-layer Neural Network

* Metrics: ROC-AUC ~0.87–0.90, Accuracy ~88–90%

* Training Notebook: [Link to Kaggle/Jupyter Notebook]

* Dataset Source: Kaggle Playground Series S5E11


💻 Local Development

* Clone the repository:
git clone :https://github.com/Kaushalya13/Predicting-Loan-Payback <br/>
cd loan-payback-predictor

* Install dependencies: <br/>
pip install -r requirements.txt

* Run the app: <br/>
python app.py

* Access at http://127.0.0.1:5000

📝 Project Structure

loan-payback-predictor/ <br/>
├── app.py                  # Flask back-end <br/>
├── index.html              # HTML front-end <br/>
├── predicting-loan-payback-notebook.ipynb  # EDA, training,submissions <br/>
├── results/                # Model artifacts (pkl files) <br/>
├── data/                   # Kaggle CSVs <br/>
├── report/                 # Assignment PDF <br/>
└── README.md               # Summary of the project


