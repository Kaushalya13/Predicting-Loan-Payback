<h1>💳 Loan Payback Predictor</h1>

Loan Payback Predictor is a deep learning–based application that predicts loan repayment outcomes using neural networks.
It analyzes borrower data such as income, credit score, and debt ratio to estimate the probability of loan payback.
This project was developed for the Kaggle Playground Series S5E11 competition.

🔗 GitHub Repository:
https://github.com/Kaushalya13/Predicting-Loan-Payback

📊 Kaggle Competition:
https://www.kaggle.com/competitions/playground-series-s5e11

✨ Key Features

* Binary Classification
Predicts loan payback (1 = Paid Back, 0 = Default) with probability scores and risk tiers.

* Feature Engineering
Custom features such as loan_to_income_ratio and monthly_debt_burden.

* Real-Time Web Application
Interactive Flask-based web form for live predictions.

* Exploratory Data Analysis (EDA)
Visual insights including distributions, correlations, and outlier detection.

* Ethical AI Considerations
Careful handling of sensitive attributes like gender and employment status.

🚀 How It Works

Data Input: Enter borrower details (annual income, credit score, etc.) via web form.
Preprocessing: Encode categoricals, scale numerics, add engineered features.
Inference: Neural network predicts probability; maps to risk level with color feedback.
Output: Displays payback chance (e.g., 78%) and assessment (e.g. Low Risk).

🛠️ Tech Stack

* Machine Learning: TensorFlow / Keras (Neural Networks)

* Data Processing: pandas, numpy, scikit-learn

* Web Framework: Flask (Back-end), HTML & CSS (Front-end)

* Visualization: matplotlib, seaborn

* Model Persistence: joblib

🧠 Research & Data Assets

* This project is for CIS6005 Computational Intelligence module, focusing on deep learning for financial prediction.

* Algorithm: Multi-layer Neural Network

* Metrics: ROC-AUC ~0.87–0.90, Accuracy ~88–90%

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

🗂️ Project Structure

loan-payback-predictor/ <br/>
├── app.py                  # Flask back-end <br/>
├── index.html              # HTML front-end <br/>
├── predicting-loan-payback-notebook.ipynb  # EDA, training,submissions <br/>
├── results/                # Model artifacts (pkl files) <br/>
├── data/                   # Kaggle CSVs <br/>
├── report/                 # Assignment PDF <br/>
└── README.md               # Summary of the project


