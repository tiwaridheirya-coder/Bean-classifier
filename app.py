import streamlit as st
import numpy as np
import joblib

# load model & scaler
model = joblib.load("bean_model.pkl")
scaler = joblib.load("scaler.pkl")

# title
st.title("🌱 Dry Bean Classification App")

st.write("Enter all bean features below:")

# input fields
Area = st.number_input("Area")
Perimeter = st.number_input("Perimeter")
MajorAxisLength = st.number_input("Major Axis Length")
MinorAxisLength = st.number_input("Minor Axis Length")
AspectRation = st.number_input("Aspect Ration")
Eccentricity = st.number_input("Eccentricity")
ConvexArea = st.number_input("Convex Area")
EquivDiameter = st.number_input("Equivalent Diameter")
Extent = st.number_input("Extent")
Solidity = st.number_input("Solidity")
roundness = st.number_input("Roundness")
Compactness = st.number_input("Compactness")
ShapeFactor1 = st.number_input("Shape Factor 1")
ShapeFactor2 = st.number_input("Shape Factor 2")
ShapeFactor3 = st.number_input("Shape Factor 3")
ShapeFactor4 = st.number_input("Shape Factor 4")

# button
if st.button("Predict"):
    
    # features array (ORDER IMPORTANT)
    features = np.array([[Area, Perimeter, MajorAxisLength, MinorAxisLength,
                          AspectRation, Eccentricity, ConvexArea, EquivDiameter,
                          Extent, Solidity, roundness, Compactness,
                          ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4]])
    
    # scaling
    scaled = scaler.transform(features)
    
    # prediction
    prediction = model.predict(scaled)
    
    # output
    st.success(f"Predicted Bean Type: {prediction[0]}")
