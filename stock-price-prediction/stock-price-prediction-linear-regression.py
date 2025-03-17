import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Fetch Stock Data
def get_stock_data(ticker, start, end):
    stock = yf.download(ticker, start=start, end=end)
    return stock[['Close']]


# Preprocess Data
def preprocess_data(stock_data):
    stock_data['Date'] = np.arange(len(stock_data))  # Convert dates into numerical values
    return stock_data

#Train Model
def train_model(data):
    X = data[['Date']]
    y = data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    error = np.sqrt(mean_squared_error(y_test, predictions))
    return model, error, X_test, y_test, predictions

# Visualize Results
def plot_results(X_test, y_test, predictions):
    plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
    plt.scatter(X_test, predictions, color='red', label='Predicted Prices')
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# Run the pipeline
ticker = 'TSLA'
start_date = '2023-01-01'
end_date = '2024-01-01'

stock_data = get_stock_data(ticker, start_date, end_date)
processed_data = preprocess_data(stock_data)
model, error, X_test, y_test, predictions = train_model(processed_data)
plot_results(X_test, y_test, predictions)

print(f'Model RMSE: {error}')

