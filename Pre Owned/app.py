import streamlit as st
import pandas as pd
import joblib

# Load the pipeline
pipeline = joblib.load('price_prediction_pipeline.pkl')

# Streamlit App
st.title("Car Price Prediction App")
st.write("Enter the car details below to get a price prediction.")

# User input
year = st.number_input("Year of Manufacture", min_value=1980, max_value=2024, step=1, value=2010)
cylinders = st.number_input("Cylinders of the Car", min_value=3, max_value=6, step=1, value=3)
odometer = st.number_input("Odometer Reading (in miles)", min_value=0, max_value=500000, step=100, value=50000)
condition = st.selectbox("Condition of the Car", ['like new', 'good', 'excellent', 'fair', 'new', 'salvage'])
fuel = st.selectbox("Enter Fuel Type", ['gas', 'diesel', 'electric', 'hybrid', 'other'])
title_status = st.selectbox("Select Title Status",  ['clean', 'rebuilt', 'salvage', 'lien', 'missing', 'parts only'])
transmission = st.selectbox("Select Transmission ", ['automatic', 'manual', 'other'])
drive = st.selectbox("Select Drive ",  ['4wd', 'rwd', 'fwd'])
size = st.selectbox("Select Seat Size", ['full-size', 'mid-size', 'sub-compact', 'compact'])
type = st.selectbox("Select Type",['offroad', 'sedan', 'pickup', 'convertible', 'van', 'truck', 'SUV',
                                   'coupe','hatchback', 'mini-van', 'wagon', 'other', 'bus'])

# Predict button
if st.button("Predict Price"):
    # Prepare user input for prediction
    user_input = pd.DataFrame({
        'year': [year],
        'cylinders': [cylinders],
        'odometer': [odometer],
        'condition': [condition],
        'fuel' : [fuel],
        'title_status':[title_status],
        'transmission':[transmission],
        'drive':[drive],
        'size':[size],
        'type':[type]
    })
    
    # Predict price
    predicted_price = pipeline.predict(user_input)[0]
    
    # Display prediction
    st.subheader(f"The predicted price of the car is: ${predicted_price:,.2f}")
