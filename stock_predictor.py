
#Import Dependencies
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset_train = pd.read_csv('GOOG.csv')
train_data = dataset_train.iloc[:, 1:2].as_matrix()

dataset_test = pd.read_csv('test_GOOGL.csv')
test_data = dataset_test.iloc[:, 1:2].as_matrix()

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
train_data = sc.fit_transform(train_data)
test_data = sc.transform(test_data)

#Data PreProcessing
X_train = []
y_train = []

for i in range(60, 3471):
    X_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    
X_train = np.array(X_train)
y_train = np.array(y_train)
    
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Defining Model Architecture
from keras.models import Sequential
from keras.layers import Dropout, Dense, LSTM

model = Sequential()
model.add(LSTM(128, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.25))
model.add(LSTM(128, return_sequences = True))
model.add(Dropout(0.25))
model.add(LSTM(64, return_sequences = True))
model.add(Dropout(0.25))
model.add(LSTM(64, return_sequences = False))
model.add(Dropout(0.25))
model.add(Dense(1, activation = 'relu'))
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])

#Training Model
model.fit(X_train, y_train, batch_size = 128, epochs = 200, verbose = 1, validation_split = 0.1)

#Saving Model
model.save('my_model.h5')


#Testing Model
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)

inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_train.shape[1], 1))
predicted_price = model.predict(X_test)
predicted_price = sc.inverse_transform(predicted_price)

data = pd.read_csv('test_GOOGL.csv')
real_data = data.iloc[:, 1:2].values
del data

plt.plot(real_data, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
