import streamlit as st
import joblib
import pandas as pd
model = joblib.load('linear_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Student Test Score Predictor')
st.write('Enter the number of hours studied to predict the test score')

hours = st.number_input('Hours Studied:', min_value = 0.0 , step = 1.0)

if st.button('Predict'):
    try:
        data = [[hours]]
        scaled_data = scaler.transform(data)
        prediction = model.predict(scaled_data)
        st.write(f'Predicted test score: {prediction[0]:.2f}')
    except Exception as e:
        st.error(f'Error {e}')
        
