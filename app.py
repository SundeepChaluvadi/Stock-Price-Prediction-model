import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load model
model = load_model('Stock Predictions Model.keras')

st.header('Stock Market Predictor')

# Stock input
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

# Download data
data = yf.download(stock, start=start, end=end)
st.subheader('Stock Data')
st.write(data)

# Train-test split
data_train = pd.DataFrame(data.Close[:int(len(data)*0.8)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.8):])

# Scaling
scaler = MinMaxScaler(feature_range=(0,1))
past_100_days = data_train.tail(100)
data_test_scaled = scaler.fit_transform(pd.concat([past_100_days, data_test], ignore_index=True))

# Plots
st.subheader('Price vs MA50')
ma_50 = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50, 'r', label='MA50')
plt.plot(data.Close, 'g', label='Close')
plt.legend()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100 = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50, 'r', label='MA50')
plt.plot(ma_100, 'b', label='MA100')
plt.plot(data.Close, 'g', label='Close')
plt.legend()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200 = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100, 'r', label='MA100')
plt.plot(ma_200, 'b', label='MA200')
plt.plot(data.Close, 'g', label='Close')
plt.legend()
st.pyplot(fig3)

# Prepare test data for prediction
x_test, y_test = [], []
for i in range(100, data_test_scaled.shape[0]):
    x_test.append(data_test_scaled[i-100:i])
    y_test.append(data_test_scaled[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predictions
y_pred = model.predict(x_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1,1))

# Prediction plot
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(y_test, 'g', label='Original Price')
plt.plot(y_pred, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)
