# prediction.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yfinance as yf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense


# Fetch the data and preprocess it
def fetch_and_preprocess(stock_symbol):
    name = yf.Ticker(stock_symbol)
    df = name.history(period="3mo")
    df = df.reset_index()
    dates = df['Date']
    close_prices = df['Close']

    # Convert 'Close' prices to a numpy array
    close_prices_array = np.array(close_prices).reshape(-1, 1)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close_prices = scaler.fit_transform(close_prices_array)

    return dates, close_prices, scaled_close_prices, scaler


# Function to create sequences for GRU
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i: i + seq_length]
        sequences.append(seq)
    return np.array(sequences)


# Define the GRU model
def build_model(X_train, y_train, sequence_length):
    model = Sequential()
    model.add(GRU(units=50, activation='relu', input_shape=(sequence_length, 1)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    return model


# Predict next 5 days' stock prices
def predict_stock_prices(model, scaled_close_prices, scaler, sequence_length):
    predicted_prices = []
    last_sequence = scaled_close_prices[-sequence_length:].reshape(1, sequence_length, 1)
    for _ in range(10):  # Change value based on requirement if for 5 days then replace 10 with 5
        prediction = model.predict(last_sequence)
        predicted_prices.append(prediction[0, 0])  # Extracting the predicted value
        last_sequence = np.append(last_sequence[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

    # Inverse transform the predicted prices
    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

    return predicted_prices.flatten()
