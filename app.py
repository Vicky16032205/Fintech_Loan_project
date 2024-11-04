%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(page_title="LoanWise Advisor", page_icon="💰", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
def generate_dummy_data(n=1000):
    np.random.seed(42)
    data = pd.DataFrame({
        'LoanID': range(1, n+1),
        'Age': np.random.randint(18, 70, n),
        'Income': np.random.randint(20000, 200000, n),
        'LoanAmount': np.random.randint(5000, 500000, n),
        'CreditScore': np.random.randint(300, 850, n),
        'MonthsEmployed': np.random.randint(0, 480, n),
        'NumCreditLines': np.random.randint(0, 20, n),
        'InterestRate': np.random.uniform(2, 15, n),
        'LoanTerm': np.random.choice([12, 36, 60, 120, 180, 240, 360], n),
        'DTIRatio': np.random.uniform(0, 0.6, n),
        'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n),
        'EmploymentType': np.random.choice(['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], n),
        'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n),
        'HasMortgage': np.random.choice([0, 1], n),
        'HasDependents': np.random.choice([0, 1], n),
        'LoanPurpose': np.random.choice(['Home', 'Auto', 'Business', 'Education', 'Personal'], n),
        'HasCoSigner': np.random.choice([0, 1], n),
        'Default': np.random.choice([0, 1], n, p=[0.7, 0.3])  # 30% default rate for demonstration
    })
    return data

@st.cache_resource
def train_model():
    data = generate_dummy_data()
    X = data.drop(['LoanID', 'Default', 'InterestRate'], axis=1)
    y = data['Default']

    # Define preprocessing for numeric columns (scale them)
    numeric_features = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'LoanTerm', 'DTIRatio']
    numeric_transformer = StandardScaler()

    # Define preprocessing for categorical columns (encode them)
    categorical_features = ['Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Create a preprocessing and modeling pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Fit the model
    model.fit(X, y)
    return model

# Train model
model = train_model()

# Sidebar inputs
st.sidebar.title("Your Loan Application")

age = st.sidebar.number_input('Age', min_value=18, max_value=100, value=30)
income = st.sidebar.number_input('Annual Income ($)', min_value=0, max_value=1000000, value=50000)
loan_amount = st.sidebar.number_input('Desired Loan Amount ($)', min_value=1000, max_value=1000000, value=100000)
credit_score = st.sidebar.slider('Credit Score', 300, 850, 650)
months_employed = st.sidebar.number_input('Months Employed', min_value=0, max_value=600, value=60)
num_credit_lines = st.sidebar.number_input('Number of Credit Lines', min_value=0, max_value=30, value=3)
loan_term = st.sidebar.selectbox('Loan Term (Months)', [12, 36, 60, 120, 180, 240, 360])
dti_ratio = st.sidebar.slider('Debt-to-Income Ratio', 0.0, 1.0, 0.3, format="%.2f")
education = st.sidebar.selectbox('Education', ['High School', 'Bachelor', 'Master', 'PhD'])
employment_type = st.sidebar.selectbox('Employment Type', ['Full-time', 'Part-time', 'Self-employed', 'Unemployed'])
marital_status = st.sidebar.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
has_mortgage = st.sidebar.checkbox('Has Mortgage')
has_dependents = st.sidebar.checkbox('Has Dependents')
loan_purpose = st.sidebar.selectbox('Loan Purpose', ['Home', 'Auto', 'Business', 'Education', 'Personal'])
has_cosigner = st.sidebar.checkbox('Has Co-Signer')

# Main content
st.title("🏦 LoanWise Advisor")
st.markdown("Your comprehensive guide to loan eligibility and financial planning")

# Predict loan eligibility
user_data = pd.DataFrame({
    'Age': [age],
    'Income': [income],
    'LoanAmount': [loan_amount],
    'CreditScore': [credit_score],
    'MonthsEmployed': [months_employed],
    'NumCreditLines': [num_credit_lines],
    'LoanTerm': [loan_term],
    'DTIRatio': [dti_ratio],
    'Education': [education],
    'EmploymentType': [employment_type],
    'MaritalStatus': [marital_status],
    'HasMortgage': [int(has_mortgage)],
    'HasDependents': [int(has_dependents)],
    'LoanPurpose': [loan_purpose],
    'HasCoSigner': [int(has_cosigner)]
})

probability = model.predict_proba(user_data)[0][1]
eligibility = "High Risk" if probability > 0.5 else "Low Risk"

# Loan Eligibility Section
st.header("🎯 Loan Risk Assessment")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Risk Status", eligibility, delta=None, delta_color="inverse")
with col2:
    st.metric("Default Probability", f"{probability:.2%}", delta=None, delta_color="inverse")
with col3:
    # Simplified interest rate calculation
    base_rate = 5.0
    risk_premium = probability * 10  # Simplified: higher risk, higher premium
    estimated_rate = base_rate + risk_premium
    st.metric("Estimated Interest Rate", f"{estimated_rate:.2f}%", delta=None, delta_color="normal")

# Personalized Financial Advice
st.header("💡 Personalized Financial Advice")
advice = []
if credit_score < 700:
    advice.append("Work on improving your credit score to reduce loan risk and potentially get better terms.")
if dti_ratio > 0.4:
    advice.append("Consider reducing your debt-to-income ratio to improve your financial health.")
if months_employed < 24:
    advice.append("A longer employment history can improve your loan prospects.")
if not has_cosigner and probability > 0.3:
    advice.append("Consider getting a co-signer to potentially reduce your loan risk.")
if num_credit_lines > 10:
    advice.append("Having many credit lines might be seen as a risk. Consider consolidating some of your credit.")
if not advice:
    advice.append("Great job! Your financial profile looks strong. Keep maintaining your good habits.")

for tip in advice:
    st.info(tip)

# Loan Repayment Simulator
st.header("💰 Loan Repayment Simulator")
monthly_payment = loan_amount * (estimated_rate/100/12) / (1 - (1 + estimated_rate/100/12)**(-loan_term))

col1, col2 = st.columns(2)
with col1:
    st.metric("Estimated Monthly Payment", f"${monthly_payment:.2f}", delta=None, delta_color="normal")
with col2:
    total_interest = monthly_payment * loan_term - loan_amount
    st.metric("Total Interest Paid", f"${total_interest:.2f}", delta=None, delta_color="normal")

# Visualization
fig = go.Figure(data=[go.Pie(labels=['Principal', 'Total Interest'],
                             values=[loan_amount, total_interest],
                             hole=.3)])
fig.update_layout(title_text="Loan Breakdown")
st.plotly_chart(fig)

# Credit Score Analysis
st.header("📊 Credit Score Analysis")
fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = credit_score,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Credit Score"},
    gauge = {
        'axis': {'range': [300, 850]},
        'bar': {'color': "darkblue"},
        'steps' : [
            {'range': [300, 580], 'color': "red"},
            {'range': [580, 670], 'color': "orange"},
            {'range': [670, 740], 'color': "yellow"},
            {'range': [740, 800], 'color': "lightgreen"},
            {'range': [800, 850], 'color': "green"}],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': credit_score}}))
st.plotly_chart(fig)

# Financial Overview Dashboard
st.header("📈 Financial Overview Dashboard")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Annual Income", f"${income:,}", delta=None, delta_color="normal")
    st.metric("Credit Score", credit_score, delta=None, delta_color="normal")
    st.metric("Loan Term", f"{loan_term} months", delta=None, delta_color="normal")
with col2:
    st.metric("Loan Amount", f"${loan_amount:,}", delta=None, delta_color="normal")
    st.metric("Debt-to-Income Ratio", f"{dti_ratio:.2%}", delta=None, delta_color="normal")
    st.metric("Employment Duration", f"{months_employed} months", delta=None, delta_color="normal")
with col3:
    st.metric("Number of Credit Lines", num_credit_lines, delta=None, delta_color="normal")
    st.metric("Estimated Monthly Payment", f"${monthly_payment:.2f}", delta=None, delta_color="normal")
    st.metric("Education Level", education, delta=None, delta_color="normal")

st.markdown("---")
st.markdown("*Note: This is a simplified model for demonstration purposes. Actual loan decisions involve many more factors and complex calculations.*")
