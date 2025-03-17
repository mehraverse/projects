import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense

# Fetch Stock Data
def get_stock_data(ticker, start, end):
    stock = yf.download(ticker, start=start, end=end)
    return stock[['Close']]

# Preprocess Data
def preprocess_data(stock_data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_data['Scaled_Close'] = scaler.fit_transform(stock_data[['Close']])
    return stock_data, scaler

# Create Sequences
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        labels.append(data[i+seq_length])
    return np.array(sequences), np.array(labels)

# Train Model Definition
def train_lstm_model(X_train, y_train):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=1, epochs=10)
    return model

# Run the pipeline 
ticker = 'AAPL'
start_date = '2023-01-01'
end_date = '2024-01-01'
seq_length = 50

stock_data = get_stock_data(ticker, start_date, end_date)
processed_data, scaler = preprocess_data(stock_data)

# Convert data into sequences
X, y = create_sequences(processed_data['Scaled_Close'].values, seq_length)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split data into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train the LSTM model
model = train_lstm_model(X_train, y_train)

# Predict stock prices
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Step 5: Visualize Results
plt.plot(y_test, color='blue', label='Actual Prices')
plt.plot(y_pred, color='red', label='Predicted Prices')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
