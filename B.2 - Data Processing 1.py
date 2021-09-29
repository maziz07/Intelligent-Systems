import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer



DATA_SOURCE = "yahoo"
COMPANY = "FB"
PRICE_VALUE = "Adj Close"
PREDICTION_DAYS = 60

def load_and_process_data(source = DATA_SOURCE, company = COMPANY, column = PRICE_VALUE, days = PREDICTION_DAYS,
        split_by_dates = True, train_size = 0.80):
    
    #Allowing to specify dates
    START = dt.datetime.strptime(input("Enter Start Date(YYYY-MM-DD): "), "%Y-%m-%d")     # Start date to read
    END = dt.datetime.strptime(input("Enter End Date(YYYY-MM-DD): "), "%Y-%m-%d")         # End date to read

    data = web.DataReader(company, source, START, END) 

    #Saving the downloaded data with user's consent
    consent = '0'
    while consent not in 'yYnN':
        consent = input("Do you want to save the downloaded data? (y/n) : ")
        if consent == 'y':
            data.to_csv(f'{company}.csv')

    required_data = data[PRICE_VALUE].values.reshape(-1,1)
    
    #Handeling NAN values
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean') 
    imputer.fit(required_data) 
    required_data = imputer.transform(required_data)

    #Feature Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(required_data) 

    X, Y = [], []
    for i in range(days, len(scaled_data)):
        X.append(scaled_data[i-days:i])
        Y.append(scaled_data[i])

    X,Y = np.array(X), np.array(Y)

    #Splitting on basis of dates or randomly
    if split_by_dates:
        x_train, x_test = X[:int(len(X)*train_size)], X[int(len(X)*train_size):]
        y_train, y_test = Y[:int(len(Y)*train_size)], Y[int(len(Y)*train_size):]
    else:
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 1 - train_size, random_state = 42)
    
    return x_train, x_test, y_train, y_test, scaler

x_train, x_test, y_train, y_test, scaler = load_and_process_data()
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


model = Sequential() # Basic neural network

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1)) 
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)


#------------------------------------------------------------------------------
# Make predictions on test data
#------------------------------------------------------------------------------

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
actual_prices = scaler.inverse_transform(y_test)


plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

#------------------------------------------------------------------------------
# Make predictions on Real data
#------------------------------------------------------------------------------
END = dt.datetime.now()
START = END - dt.timedelta(days = PREDICTION_DAYS * 2)

data = web.DataReader(COMPANY, DATA_SOURCE, START, END) 
required_data = data[PRICE_VALUE].values.reshape(-1,1)
scaled_data = scaler.transform(required_data) 

X = []
for i in range(PREDICTION_DAYS, len(scaled_data)):
    X.append(scaled_data[i-PREDICTION_DAYS:i])

X = np.array(X)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

prediction = model.predict(X)
prediction = scaler.inverse_transform(prediction)
print(f'Prediction : {prediction[-1]}')