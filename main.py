import pdb

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import mplfinance as mpl
import plotly.graph_objects as go

from pandas.plotting import lag_plot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional

# Load data
company = "AMZN"
start = dt.datetime(2019, 1, 1)
end = dt.datetime.now()

data = web.DataReader(company, 'yahoo', start, end)

# Prepare data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
scaled_data = scaled_data[:, 0]
prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x])
    y_train.append(scaled_data[x])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# Extending the model to include RSI indicator
# ---------------------------------------------------------------------------------------
def Plot_RSI(days):
    delta = data['Adj Close'].diff(1)
    delta.dropna(inplace=True)

    positive = delta.copy()
    negative = delta.copy()

    positive[positive < 0] = 0
    negative[negative > 0] = 0

    days = days  # default is 14

    average_gain = positive.rolling(window=days).mean()
    average_loss = abs(negative.rolling(window=days).mean())

    relative_strength = average_gain / average_loss
    RSI = 100.0 - (100.0 / (1.0 + relative_strength))

    combined = pd.DataFrame()
    combined['Adj Close'] = data['Adj Close']
    combined['RSI'] = RSI

    # plotting the Adjusted Close value
    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(211)
    ax1.plot(combined.index, combined['Adj Close'], color='lightgray')
    ax1.set_title("Adjusted Close Price", color='white')

    ax1.grid(True, color='#555555')
    ax1.set_axisbelow(True)
    ax1.set_facecolor('black')
    ax1.figure.set_facecolor('#121212')
    ax1.tick_params(axis='both', colors='white')

    # plotting the RSI indicator
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(combined.index, combined['RSI'], color='lightgray')
    ax2.axhline(0, linestyle='--', alpha=0.5, color='#ff0000')
    ax2.axhline(10, linestyle='--', alpha=0.5, color='#ffaa00')
    ax2.axhline(20, linestyle='--', alpha=0.5, color='#00ff00')
    ax2.axhline(30, linestyle='--', alpha=0.5, color='#cccccc')
    ax2.axhline(70, linestyle='--', alpha=0.5, color='#cccccc')
    ax2.axhline(80, linestyle='--', alpha=0.5, color='#00ff00')
    ax2.axhline(90, linestyle='--', alpha=0.5, color='#ffaa00')
    ax2.axhline(100, linestyle='--', alpha=0.5, color='#ff0000')

    ax2.set_title("RSI Value")
    ax2.grid(False)
    ax2.set_axisbelow(True)
    ax2.set_facecolor('black')
    ax2.tick_params(axis='both', colors='white')

    plt.show()


Plot_RSI(14)
# ---------------------------------------------------------------------------------------

# build the model
# ---------------------------------------------------------------------------------------
# model = Sequential()
#
# model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# model.add(Dropout(0.2))
# model.add(LSTM(units=50, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(units=50))
# model.add(Dropout(0.2))
# model.add(Dense(units=1))  # Prediction of the next closest price
#
# model.compile(optimizer='adam', loss='mean_squared_error')
# model.fit(x_train, y_train, epochs=3, batch_size=32)


# --------------------------------------------------------------------------------------
def create_model(sequence_length, n_features, units=50, cell=LSTM, n_layers=2, dropout=0.2,
                 loss='mean_squared_error', optimizer='adam', bidirectional=False):
    model = Sequential()

    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True),
                                        batch_input_shape=(None, sequence_length, n_features)))
            else:
                model.add(cell(units, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # Hidden layers
            if bidirectional:
                model.add(bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        model.add(Dropout(dropout))
    model.add(Dense(units))
    model.compile(optimizer=optimizer, loss=loss)
    return model


_model = create_model(50, 1)

_model.fit(x_train, y_train, epochs=3, batch_size=32)
# ----------------------------------------------------------------------------------
# Test the model accuracy on existing data
# Load test data
test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

test_data = web.DataReader(company, 'yahoo', test_start, test_end)
test_data = test_data[1:]
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

# Make predictions on test data
x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# -----------------------------------------------------------------------
train_data, test_data = data[0:int(len(data) * 0.7)], data[int(len(data) * 0.7):]

training_data = train_data['Close'].values
test_data = test_data['Close'].values

model_predictions = []
history = [x for x in training_data]
N_test_observations = len(test_data)

for time_point in range(N_test_observations):
    model = ARIMA(history, order=(1, 1, 0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    model_predictions.append(yhat)
    true_test_value = test_data[time_point]
    history.append(true_test_value)

MSE_error = mean_squared_error(test_data, model_predictions)
print('Testing Mean Squared Error is {}'.format(MSE_error))
# ------------------------------------------------------------------------

# -----------------------------------------------------------------------
test_set_range = data[int(len(data) * 0.7):].index

plt.plot(test_set_range, model_predictions, color='blue', marker='o',
         linestyle='dashed', label='Predicted Price')

plt.plot(test_set_range, test_data, color='red', label='Actual Price')

plt.title('Facebook Price Prediction')
plt.xlabel('Date')
plt.ylabel('Prices')
plt.legend()
plt.show()
# ------------------------------------------------------------------------

predicted_prices = _model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)


# plot the test predictions
# plt.plot(actual_prices, color="black", label=f"Actual {company} price")
# plt.plot(predicted_prices, color="green", label=f"Predicted {company} price")
# plt.title(f"{company} share price")
# plt.xlabel('Time')
# plt.ylabel(f'{company} Share price')
# plt.legend()
# plt.show()
# --------------------------------------
def plot_graph(avg_window):
    avg = data['Close'].rolling(window=avg_window, min_periods=1).mean()
    trace1 = {
        'x': data.index,
        'open': data["Open"],
        'close': data["Close"],
        'high': data['High'],
        'low': data['Low'],
        'type': 'candlestick',
        'name': company,
        'showlegend': True
    }
    trace2 = {
        'x': data.index,
        'y': avg,
        'type': 'scatter',
        'mode': 'lines',
        'line': {
            'width': 1,
            'color': 'blue'
        },
        'name': company,
        'showlegend': True
    }
    fig = go.Figure(data=[trace1, trace2])
    fig.show()


plot_graph(90)

# -------------------------------------
# candlestick = go.Candlestick(
#     x=data.index,
#     y=avg,
#     open=data['Open'],
#     high=data['High'],
#     low=data['Low'],
#     close=data['Close'],
#     type='scatter',
#     showlegend=True
# )
# fig = go.Figure(data=[candlestick])
# fig.show()
# --------------------------------------
# Predict next day
real_data = [model_inputs[len(model_inputs) - prediction_days:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = _model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction using LSTM model: {prediction}")
print(f"Prediction using ARIMA model: {model_predictions}")

combined_prediction = (prediction[-1] + model_predictions[-1]) / 2
print(f"Prediction using ensemble approach: {combined_prediction}")

