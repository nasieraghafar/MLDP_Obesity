import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import base64

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

# Add custom background
def add_bg_from_local(image_path):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{image_path}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
 
# Encode image to base64 for background
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()
 
# Call function to add background
image_path = "Picture1.png"  # Replace with your image path
encoded_image = encode_image_to_base64(image_path)
add_bg_from_local(encoded_image)

# Add custom CSS for the container
st.markdown(
    """
    <style>
    .custom-container h1 {
        color: white; 
        font-family:  Arial Black, sans-serif; /* Modern font */
        font-size: 40px; /* Font size for title */
        margin: 0; /* Remove default margin */
        text-align: center; /* Center the text */

    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add the HTML structure for the container and text
st.markdown(
    """
    <div class="custom-container">
        <h1>Obesity Prediction Application</h1>
    </div>
    """,
    unsafe_allow_html=True
)


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

# Define the class labels
class_labels = [
    'Normal_Weight', 'Obesity_Type_I', 'Obesity_Type_II',
    'Obesity_Type_III', 'Overweight_Level_I', 'Overweight_Level_II'
]

st.markdown(
    """
    <style>
    .scrollable-table {
        background-color: rgba(255, 255, 255, 0.9); /* Set background color to white */
        padding: 10px; /* Add padding for better readability */
        border-radius: 10px; /* Rounded corners */
        border: 1px solid #ddd; /* Light border */
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); /* Subtle shadow */
        max-width: 100%; /* Prevent the table from exceeding the container width */
        max-height: 400px; /* Optional: limit the height of the table */
        overflow-x: auto; /* Enable horizontal scrolling */
        overflow-y: auto; /* Enable vertical scrolling (if needed) */
    }
    .custom-subheader {
        color: white; /* Set font color to white */
        font-size: 24px; /* Adjust font size if needed */
        font-weight: bold; /* Make it bold to resemble a subheader */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display user inputs in a scrollable table
st.markdown('<h2 class="custom-subheader">Your Input Parameters</h2>', unsafe_allow_html=True)

input_summary = {
    "Gender": gender,
    "Age": age,
    "Height (cm)": height,
    "Weight (kg)": weight,
    "Family History of Obesity": family_history,
    "Frequent High Calorie Food Consumption (FAVC)": favc,
    "Vegetable Consumption Frequency (FCVC)": fcvc,
    "Number of Meals per Day (NCP)": ncp,
    "Food Between Meals (CAEC)": caec,
    "Smoker": smoke,
    "Water Consumption (CH2O)": ch2o,
    "Monitors Calorie Intake (SCC)": scc,
    "Physical Activity Level (FAF)": faf,
    "Time Spent on Devices (TUE)": tue,
    "Alcohol Consumption (CALC)": calc,
    "Mode of Transport (MTRANS)": mtrans,
}

# Create the table and wrap it in a scrollable container
st.markdown(
    f"""
    <div class="scrollable-table">
        {pd.DataFrame([input_summary]).to_html(index=False, escape=False)}
    </div>
    """,
    unsafe_allow_html=True
)


prediction_label = prediction[0]

# Add custom CSS for the prediction text
st.markdown(
    """
    <style>
    .prediction-box {
        background-color: rgba(0, 0, 0, 0.7); /* Semi-transparent black background */
        color: white; /* White text for contrast */
        font-size: 28px; /* Larger font size for emphasis */
        font-weight: bold; /* Make the text bold */
        text-align: center; /* Center the text */
        padding: 20px; /* Add padding for spacing inside the box */
        border-radius: 15px; /* Rounded corners */
        border: 2px solid rgba(255, 255, 255, 0.5); /* Light white border */
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.5); /* Subtle shadow for a floating effect */
        margin-top: 20px; /* Add space above the prediction box */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the prediction with the new styling
st.markdown(f'<div class="prediction-box">Prediction: {prediction_label}</div>', unsafe_allow_html=True)

