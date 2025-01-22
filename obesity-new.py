import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load('svm_obesity_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')  # Load feature names from training

ohe_columns = [
    'Gender_Female', 'Gender_Male', 'family_history_no', 'family_history_yes', 
    'FAVC_no', 'FAVC_yes', 'CAEC_no', 'CAEC_Always', 'CAEC_Frequently', 
    'CAEC_Sometimes', 'SMOKE_no', 'SMOKE_yes', 'SCC_no', 'SCC_yes', 
    'CALC_no', 'CALC_Always', 'CALC_Frequently', 'CALC_Sometimes', 
    'MTRANS_Automobile', 'MTRANS_Bike', 'MTRANS_Motorbike', 
    'MTRANS_Public_Transportation', 'MTRANS_Walking'
]

numerical_inputs = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

# Function to preprocess input data (OHE and scaling)
def preprocess_input_data(inputs):
    # Convert categorical inputs to one-hot encoded columns
    data = pd.DataFrame([inputs])

    # Ensure all one-hot encoded columns are present and fill missing ones with 0
    for col in ohe_columns:
        if col not in data.columns:
            data[col] = 0
    
    # Reorder the columns to match the order of one-hot encoded columns in training
    data = data[ohe_columns]

    # Now add the numerical features (ensure they are in the correct order)
    numerical_values = [inputs[col] for col in numerical_inputs]
    numerical_data = pd.DataFrame([numerical_values], columns=numerical_inputs)
    
    # Combine both categorical and numerical features
    final_data = pd.concat([data, numerical_data], axis=1)

    # Scale numerical features (use the fitted scaler)
    # final_data[numerical_inputs] = scaler.transform(final_data[numerical_inputs])

    # Ensure the final data has the same feature order as the model expects
    final_data = final_data[feature_names]

    return final_data

# Streamlit UI
st.title('Obesity Prediction Model')

with st.sidebar:
    # Input fields
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 70, 25)
    height = st.slider("Height (in cm)", 120, 200, 150)
    weight = st.slider("Weight (in kg)", 30, 200, 70)
    family_history = st.selectbox("Family History of Obesity", ["No", "Yes"])
    favc = st.selectbox("Frequent Consumption of High Calorie Food (FAVC)", ["No", "Yes"])
    fcvc = st.slider("Frequent Consumption of Vegetables (FCVC)", 1, 3, 2)
    ncp = st.slider("Number of Meals per Day (NCP)", 1, 5, 3)
    caec = st.selectbox("Food between Meals (CAEC)", ["No", "Sometimes", "Frequently", "Always"])
    smoke = st.selectbox("Smoker", ["No", "Yes"])
    ch2o = st.slider("Water Consumption (CH2O)", 1, 3, 2)
    scc = st.selectbox("Monitor Calorie Intake (SCC)", ["No", "Yes"])
    faf = st.slider("Physical Activity Level (FAF)", 0, 3, 1)
    tue = st.slider("Time Level Spent on Devices (TUE)", 0, 2, 1)
    calc = st.selectbox("Drink Alcohol? (CALC)", ["No", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox("Mode of Transport (MTRANS)", ["Public Transport", "Walking", "Automobile", "Motorbike", "Bike"])



# Prepare input data
inputs = {
    'Age': age,
    'Height': height / 100,
    'Weight': weight,
    'FCVC': fcvc,
    'NCP': ncp,
    'CH2O': ch2o,
    'FAF': faf,
    'TUE': tue,
    'Gender_Female': True if gender == "Female" else False,
    'Gender_Male': True if gender == "Male" else False,
    'family_history_no': True if family_history == "No" else False,
    'family_history_yes': True if family_history == "Yes" else False,
    'FAVC_no': True if favc == "No" else False,
    'FAVC_yes': True if favc == "Yes" else False,
    'CAEC_Always': True if caec == "Always" else False,
    'CAEC_Frequently': True if caec == "Frequently" else False,
    'CAEC_Sometimes': True if caec == "Sometimes" else False,
    'CAEC_no': True if caec == "No" else False,
    'SMOKE_no': True if smoke == "No" else False,
    'SMOKE_yes': True if smoke == "Yes" else False,
    'SCC_no': True if scc == "No" else False,
    'SCC_yes': True if scc == "Yes" else False,
    'CALC_Always': True if calc == "Always" else False,
    'CALC_Frequently': True if calc == "Frequently" else False,
    'CALC_Sometimes': True if calc == "Sometimes" else False,
    'CALC_no': True if calc == "No" else False,
    'MTRANS_Automobile': True if mtrans == "Car" else False,
    'MTRANS_Bike': True if mtrans == "Bike" else False,
    'MTRANS_Motorbike': True if mtrans == "Motorbike" else False,
    'MTRANS_Public_Transportation': True if mtrans == "Public_Transportation" else False,
    'MTRANS_Walking': True if mtrans == "Walking" else False
}

# Preprocess input data
processed_data = preprocess_input_data(inputs)

# Make prediction using the model
prediction = model.predict(processed_data)

# Log the prediction
st.write("Model Prediction:", prediction)

# Define the class labels
class_labels = [
    'Normal_Weight', 'Obesity_Type_I', 'Obesity_Type_II',
    'Obesity_Type_III', 'Overweight_Level_I', 'Overweight_Level_II'
]
prediction_label = prediction[0]

# Display the result
st.subheader(f"Prediction: {prediction_label}")
