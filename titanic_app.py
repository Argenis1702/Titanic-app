 
import pandas as pd
import numpy as np
import pickle
import streamlit as st

# Load the trained model
import os
import pickle

# Get the absolute path of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "titanic_model.pkl")

# Load the model with error handling
try:
    with open(model_path, "rb") as file:
        model, scaler = pickle.load(file)
    print("Model loaded successfully")
except FileNotFoundError:
    print(f"Error: The file {model_path} was not found")


# Streamlit App
def main():
    st.title("Titanic Survival Prediction")
    st.write("Enter passenger details to predict survival.")

    pclass = st.selectbox("Passenger Class (1 = First, 2 = Second, 3 = Third)", [1, 2, 3])
    sex = st.radio("Sex", ["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    fare = st.number_input("Fare", min_value=0.0, value=50.0)

    embarked = st.selectbox("Embarked (Port of Boarding)", ["C", "Q", "S"])
    embarked_dict = {"C": 0, "Q": 1, "S": 2}
    sex_dict = {"Male": 0, "Female": 1}

    # Convert input to model format
    features = np.array([[pclass, sex_dict[sex], age, fare, embarked_dict[embarked]]])
    features_scaled = scaler.transform(features)

    if st.button("Predict"):
        prediction = model.predict(features_scaled)
        result = "Survived" if prediction[0] == 1 else "Did Not Survive"
        st.success(f"The prediction is: {result}")

if __name__ == "__main__":
    main()
