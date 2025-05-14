import streamlit as st
import joblib
import numpy as np
st.title('Tesla Stock Prediction Model')
st.markdown('---')
st.header('How its Implemented')
st.write('''
The XGBoost Regressor model is used to predict 
stock prices according to volume,open price,close price,high price and low price
''')
col1,col2=st.columns(2)
with col1:
    st.image('tesla.jpeg')
with col2:
    st.header('Stock Prediction Using XGBoost Regressor Model')

# Load the saved model
model = joblib.load("xgb.pkl")  # Replace with your actual model file

st.markdown('---')

# Title
st.title("📈 Adjusted Close Price Predictor")

st.write("Enter stock values below to predict the Adjusted Close price:")

# Input fields
open_price = st.number_input("Enter The Open Price", min_value=0.0, value=100.0)
high_price = st.number_input("Enter The High Price", min_value=0.0, value=105.0)
low_price = st.number_input("Enter The Low Price", min_value=0.0, value=95.0)
close_price = st.number_input("Enter The Close Price", min_value=0.0, value=102.0)
volume = st.number_input("The Amount Of Stocks sold in the Day(Volume)", min_value=0.0, value=1000000.0, step=1000.0)

st.markdown('---')
st.header('What is the Adjusted Close Price?')
st.write('''
Adjusted Close is the stock’s closing price after accounting for dividends, stock splits, 
and other corporate actions—giving a more accurate reflection of the stock's true value over time.
''')
st.markdown('---')
# Predict button
if st.button("Predict Adjusted Close"):
    input_features = np.array([[high_price, close_price, low_price, volume, open_price]])  # must match your training order
    prediction = model.predict(input_features)
    st.success(f"📊 Predicted Adjusted Close Price: **${prediction[0]:.2f}**")
