import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD

df = pd.read_csv('D:/PlantFactory/Basil202107.csv')
df = df.sort_values('Date').reset_index(drop=True)

df['Temp'] = df['Temp'].astype(float)

print(df.shape)

plt.figure(figsize=(20,7))
plt.plot(df['Date'].values, df['Temp'].values, label = 'Temp', color = 'red')
plt.xticks(np.arange(100,df.shape[0],200))
plt.xlabel('Date')
plt.ylabel('Temp (oC)')
plt.legend()
plt.savefig('D:/PlantFactory/Temp.png', dpi=300)
plt.show()

#
num_shape = 200
# num_shape = 1900

train = df.iloc[:num_shape, 1:2].values
test = df.iloc[num_shape:, 1:2].values

print(train)

sc = MinMaxScaler(feature_range = (0, 1))
train_scaled = sc.fit_transform(train)

X_train = []

#Price on next day
y_train = []

# Now we take one row and cut it with a window of 60 elements
window = 60   #60

for i in range(window, num_shape):
    X_train_ = np.reshape(train_scaled[i-window:i, 0], (window, 1))
    X_train.append(X_train_)
    y_train.append(train_scaled[i, 0])
X_train = np.stack(X_train)
y_train = np.stack(y_train)

# Initializing the Recurrent Neural Network
model = Sequential()
#Adding the first LSTM layer with a sigmoid activation function and some Dropout regularization
#Units - dimensionality of the output space
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
# Adding the output layer
model.add(Dense(units = 1))
model.summary()

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
# history=model.fit(X_train, y_train, epochs = 10, batch_size = 32);
history=model.fit(X_train, y_train, epochs = 300, batch_size = 16)  #epochs=300
# N_epochs = np.array([0:epochs])

# Prediction
df_volume = np.vstack((train, test))

inputs = df_volume[df_volume.shape[0] - test.shape[0] - window:]
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

num_2 = df_volume.shape[0] - num_shape + window

X_test = []

for i in range(window, num_2):
    X_test_ = np.reshape(inputs[i - window:i, 0], (window, 1))
    X_test.append(X_test_)

X_test = np.stack(X_test)

predict = model.predict(X_test)
predict = sc.inverse_transform(predict)
diff = predict - test

print("MSE:", np.mean(diff**2))
print("MAE:", np.mean(abs(diff)))
print("RMSE:", np.sqrt(np.mean(diff**2)))

plt.figure(figsize=(20,7))
plt.plot(df['Date'].values[0:], df_volume[0:], color = 'red', label = 'Real Temp')
plt.plot(df['Date'][-predict.shape[0]:].values, predict, color = 'blue', label = 'Predicted Temp')
plt.xticks(np.arange(100,df[0:].shape[0],200))
plt.title('Temp Prediction')
plt.xlabel('Date')
plt.ylabel('Temp(oC)')
plt.legend()
plt.savefig('D:/PlantFactory/Temp_predict.png', dpi=300)
plt.show()

# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.figure(figsize=(10,7))
# plt.subplot(1,2,1)
# plt.plot(epochs, loss, 'ro', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid()
# plt.subplot(1,2,2)
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# plt.plot(epochs, acc, 'r', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('accuracy')
# plt.legend()
# plt.grid()
# plt.savefig('D:/Price/citrus/Hallabong/hallabong_loss_acc.png', dpi=300)
# # plt.ylim((0,4))
# plt.show()

# N_loss = loss
# N_val_loss = val_loss
# N_acc = acc
# N_val_acc = val_acc
# N_echos = np.arange(0, len(acc))+1
#
# f_loss = pd.DataFrame(np.column_stack((N_echos, N_loss, N_val_loss)))
# f_acc = pd.DataFrame(np.column_stack((N_echos, N_acc, N_val_acc)))
# f_loss.to_csv('D:/Price/citrus/Hallabong/loss.csv', index=False)
# f_acc.to_csv('D:/Price/citrus/Hallabong/acc.csv', index=False)

pred_ = predict[-1].copy()
prediction_full = []
window = 60

df_copy = df.iloc[:, 1:2][1:].values

for j in range(360):
    df_ = np.vstack((df_copy, pred_))
    train_ = df_[:num_shape]
    test_ = df_[num_shape:]

    df_volume_ = np.vstack((train_, test_))

    inputs_ = df_volume_[df_volume_.shape[0] - test_.shape[0] - window:]
    inputs_ = inputs_.reshape(-1, 1)
    inputs_ = sc.transform(inputs_)

    X_test_2 = []

    for k in range(window, num_2):
        X_test_3 = np.reshape(inputs_[k - window:k, 0], (window, 1))
        X_test_2.append(X_test_3)

    X_test_ = np.stack(X_test_2)
    predict_ = model.predict(X_test_)
    pred_ = sc.inverse_transform(predict_)
    prediction_full.append(pred_[-1][0])
    df_copy = df_[j:]
prediction_full_new = np.vstack((predict, np.array(prediction_full).reshape(-1, 1)))

df_date = df[['Date']]

for h in range(360):
    df_date_add = pd.to_datetime(df_date['Date'].iloc[-1]) + pd.DateOffset(hours=1)   #days=1
    df_date_add = pd.DataFrame([df_date_add.strftime("%Y-%m-%d %H")], columns=['Date'])
    df_date = df_date.append(df_date_add)
df_date = df_date.reset_index(drop=True)

plt.figure(figsize=(20,7))
plt.plot(df['Date'].values[0:], df_volume[0:], color = 'red', label = 'Temp')
plt.plot(df_date['Date'][-prediction_full_new.shape[0]:].values, prediction_full_new, color = 'blue', label = 'Predicted Temp')
plt.xticks(np.arange(100,df[0:].shape[0],200))
plt.title('Temp Prediction')
plt.xlabel('Date')
plt.ylabel('Temp(oC)')
plt.legend()
plt.savefig('D:/PlantFactory/predict_future.png', dpi=300)
plt.show()

# num_shape = 1500 인 경우
a = df_date['Date'][num_shape:]
b = prediction_full_new[:,0]
print(a.size)
print(b.size)
print(a.ndim)
print(b.ndim)
print(np.column_stack((a,b)))

f_data = pd.DataFrame(np.column_stack((a,b)))
f_data.to_csv('D:/PlantFactory/predict_future.csv', index=False)
