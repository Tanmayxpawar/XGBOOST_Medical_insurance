import streamlit as st
import numpy as np
from pickle import load

# Load the trained model
model = load(open('insurancemodel.pkl', 'rb'))

# Streamlit app
st.title("Insurance Charges Prediction")

st.markdown("""
    This app predicts insurance charges based on your personal information.
    Please enter the details below:
""")

# Create columns for better layout
col1, col2 = st.columns(2)

# Input fields
with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=25, help="Enter your age in years")
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0, help="Enter your Body Mass Index")
    
with col2:
    children = st.number_input("Children", min_value=0, max_value=10, value=0, help="Enter the number of children covered by health insurance")
    smoker = st.selectbox("Smoker", options=["yes", "no"], help="Select whether you smoke")

# Convert inputs to the correct format
smoker_int = 1 if smoker == 'yes' else 0

# Prepare the input for the model
input_data = np.array([[age, bmi, children, smoker_int]])

# Add a button for prediction
if st.button("Predict"):
    # Make prediction
    prediction = model.predict(input_data)
    st.success(f"Predicted Insurance Charges: ${prediction[0]:.2f}")

# Optional: Add some more information or footer
st.markdown("""
    ---
    **Note:** This prediction is based on a trained machine learning model and may not reflect actual insurance costs.
""")
