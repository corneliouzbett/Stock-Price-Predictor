from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

model = load_model('my_model.h5')

dataset_train = pd.read_csv('GOOG.csv')
train_data = dataset_train.iloc[:, 1:2].as_matrix()

dataset_test = pd.read_csv('test_GOOGL.csv')
test_data = dataset_test.iloc[:, 1:2].as_matrix()

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
train_data = sc.fit_transform(train_data)
test_data = sc.transform(test_data)

inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
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