#!/usr/bin/env python
# coding: utf-8

# In[6]:


import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time

# Load the trained model using pickle
with open("kmeans_model.pkl", "rb") as model_file:
    kmeans = pickle.load(model_file)

# Streamlit UI
st.title("🚑 AI-Powered Triage Prediction System")
st.write("Enter patient details to predict triage priority.")

# Input fields for patient details
age = st.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
chest_pain_type = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
blood_pressure = st.number_input("Blood Pressure", min_value=50, max_value=250, value=120)
cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=200)
max_heart_rate = st.number_input("Max Heart Rate", min_value=50, max_value=220, value=150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
plasma_glucose = st.number_input("Plasma Glucose", min_value=50, max_value=300, value=100)
skin_thickness = st.number_input("Skin Thickness", min_value=10, max_value=100, value=30)
insulin = st.number_input("Insulin Level", min_value=50, max_value=300, value=150)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree", min_value=0.1, max_value=2.5, value=1.0)
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("History of Heart Disease", ["No", "Yes"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
smoking_status = st.selectbox("Smoking Status", ["Non-Smoker", "Smoker", "Unknown"])

# Convert categorical inputs to numerical values
gender = 1 if gender == "Male" else 0
exercise_angina = 1 if exercise_angina == "Yes" else 0
hypertension = 1 if hypertension == "Yes" else 0
heart_disease = 1 if heart_disease == "Yes" else 0
residence_type = 1 if residence_type == "Urban" else 0
smoking_status = {"Non-Smoker": 0, "Smoker": 1, "Unknown": 2}[smoking_status]

# Create input array for prediction
input_data = np.array([
    age, gender, chest_pain_type, blood_pressure, cholesterol,
    max_heart_rate, exercise_angina, plasma_glucose, skin_thickness,
    insulin, bmi, diabetes_pedigree, hypertension, heart_disease,
    residence_type, smoking_status
]).reshape(1, -1)

# Predict triage priority
if st.button("Predict Triage Level"):
    with st.spinner("Analyzing patient data..."):
        time.sleep(2)  # Simulate processing time
        cluster = kmeans.predict(input_data)[0]  # Get cluster label

        # Map clusters to triage categories
        triage_levels = {0: "🔴 Immediate (Critical)", 1: "🟡 Urgent", 2: "🟢 Minor", 3: "⚫ Non-Emergency"}
        result = triage_levels.get(cluster, "Unknown")

        st.success(f"Triage Priority: {result}")
        st.write("This prediction is based on clustering of patient symptoms.")


# In[ ]:


get_ipython().system('jupyter nbconvert --to script app2.ipynb')


# In[ ]:




