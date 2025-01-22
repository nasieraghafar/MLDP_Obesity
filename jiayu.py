import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
 
st.title("Obesity Prediction App")
 
st.sidebar.header("User Input Parameters")
 
def user_input_features():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age = st.sidebar.slider("Age", 10, 80, 30)
    height = st.sidebar.slider("Height (in cm)", 130, 200, 170)
    weight = st.sidebar.slider("Weight (in kg)", 30, 150, 70)
    family_history = st.sidebar.selectbox("Family History of Obesity", ["Yes", "No"])
    favc = st.sidebar.selectbox("Frequent Consumption of High Caloric Food (FAVC)", ["Yes", "No"])
    smoke = st.sidebar.selectbox("Smokes?", ["Yes", "No"])
    scc = st.sidebar.selectbox("Monitor Calories (SCC)?", ["Yes", "No"])
    faf = st.sidebar.selectbox("Physical Activity (FAF)", ["Low", "Medium", "High"])
    mtrans = st.sidebar.selectbox("Mode of Transportation (MTRANS)", ["Walking", "Public_Transportation", "Automobile", "Bike", "Motorbike"])
    caec = st.sidebar.selectbox("Eating Habit (CAEC)", ["No", "Sometimes", "Frequently", "Always"])
    calc = st.sidebar.selectbox("Caloric Intake (CALC)", ["No", "Sometimes", "Frequently", "Always"])
    fcvc = st.sidebar.slider("Frequency of Consumption of Vegetables (FCVC)", 1, 3, 2)
    ncp = st.sidebar.slider("Number of Meals per Day (NCP)", 1, 5, 3)
    ch2o = st.sidebar.slider("Daily Water Consumption (CH2O in liters)", 1, 3, 2)
    tue = st.sidebar.slider("Time Using Technology (TUE in hours)", 0, 2, 1)
 
    data = {
        "Gender": gender,
        "Age": age,
        "Height": height / 100,  # Convert cm to meters
        "Weight": weight,
        "family_history": family_history,
        "FAVC": favc,
        "SMOKE": smoke,
        "SCC": scc,
        "FAF": faf,
        "MTRANS": mtrans,
        "CAEC": caec,
        "CALC": calc,
        "FCVC": fcvc,
        "NCP": ncp,
        "CH2O": ch2o,
        "TUE": tue,
    }
    features = pd.DataFrame(data, index=[0])
    return features
 
user_input = user_input_features()
 
# Display user input
st.subheader("User Input Parameters")
st.write(user_input)
 
# Preprocess the dataset
def preprocess_data(df):
    label_encodings = {
        "Gender": {"Male": 0, "Female": 1},
        "family_history": {"Yes": 1, "No": 0},
        "FAVC": {"Yes": 1, "No": 0},
        "SMOKE": {"Yes": 1, "No": 0},
        "SCC": {"Yes": 1, "No": 0},
        "FAF": {"Low": 0, "Medium": 1, "High": 2},
        "MTRANS": {"Walking": 0, "Public_Transportation": 1, "Automobile": 2, "Bike": 3, "Motorbike": 4},
        "CAEC": {"No": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
        "CALC": {"No": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
    }
    for col, mapping in label_encodings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df
 
# Preprocess user input
preprocessed_input = preprocess_data(user_input)
 
# Ensure column order matches training data
@st.cache_data
def load_model():
    return joblib.load("obesity_model.pkl")
 
try:
    # Load pre-trained model
    clf = load_model()
 
    # Ensure column order matches the model's training data
    preprocessed_input = preprocessed_input[clf.feature_names_in_]
 
    # Make predictions
    prediction = clf.predict(preprocessed_input)[0]
    prediction_proba = clf.predict_proba(preprocessed_input)[0]
 
    # Map prediction to obesity level labels
    obesity_levels = {
        "Insufficient_Weight": "Insufficient Weight",
        "Normal_Weight": "Normal Weight",
        "Overweight_Level_I": "Overweight Level I",
        "Overweight_Level_II": "Overweight Level II",
        "Obesity_Type_I": "Obesity Type I",
        "Obesity_Type_II": "Obesity Type II",
        "Obesity_Type_III": "Obesity Type III",
    }
 
    prediction_label = obesity_levels.get(prediction, "Unknown")
 
    # Display prediction
    st.subheader("Prediction")
    st.write(f"Predicted Obesity Level: {prediction_label}")
 
    # Display prediction probability
    st.subheader("Prediction Probability")
    for level, prob in zip(clf.classes_, prediction_proba):
        st.write(f"{obesity_levels.get(level, level)}: {prob * 100:.2f}%")
except Exception as e:
    st.error(f"Error: {e}")