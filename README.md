# Loan Eligibility and Financial Advisory System


# Overview
The Loan Eligibility and Financial Advisory System is a machine learning-based web application designed to assist financial institutions and users in evaluating loan eligibility. The system also offers personalized financial advice based on an applicant's profile, providing insights on factors like credit score, debt-to-income ratio, and other financial metrics.

# Features
Loan Eligibility Prediction: Predicts loan eligibility based on user financial data using a Random Forest Classifier.
Financial Advisory: Offers personalized advice to improve loan eligibility by analyzing key factors (e.g., credit score, debt-to-income ratio).
Feature Importance Analysis: Identifies and visualizes the most influential features in loan approval decisions.
User-Friendly Interface: A command-line interface for quick and effective predictions with detailed explanations.

# Tech Stack
Programming Language: Python
Libraries:
Data Processing: Pandas, NumPy
Machine Learning: scikit-learn
Visualization: Matplotlib, Seaborn
Model: Random Forest Classifier

# Installation and Setup
Prerequisites
Python (3.10 or later recommended)

Install required Python packages by running:
  pip install -r requirements.txt

Running the Project
Clone the Repository:
  git clone: https://github.com/Vicky16032205/Fintech_Loan_project.git    
  cd Fintech_Loan_project

Run the Script:
  python Fintech_Loan_project.py

# Usage
Load the dataset Loan_default.csv (replace the path if different).

Preprocess the data to handle any missing values or non-numeric columns.

Train the model on the dataset and evaluate using classification metrics.

Use the get_loan_advice() function to provide a loan eligibility decision and advisory based on applicant data.

# Evaluation
The model's performance is evaluated using:

Classification Report: Precision, recall, and F1-score to assess model accuracy.
Confusion Matrix: Visualized using Seaborn to assess prediction accuracy.
Feature Importance: Top influential features in loan eligibility.

# Future Improvements
Web Interface: Add a web-based user interface for broader accessibility.
Advanced Financial Advice: Use additional metrics for more nuanced recommendations.
Model Optimization: Experiment with other algorithms to improve prediction accuracy.


# License
This project is licensed under the MIT License. See the LICENSE file for more details.
