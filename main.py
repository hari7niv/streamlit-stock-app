import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import *
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import Sequential, layers, losses
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow import keras

stock = st.text_input('Symbol of the stock or Crypto')
start = dt.datetime(2015, 1, 1)
end = dt.datetime.now()
if stock:
     st.text('Please wait this will take max 5 min.......')
     data = web.DataReader(stock, 'yahoo', start, end)
     sclar = MinMaxScaler(feature_range=(0, 1))
     # y = sclar.fit_transform(data['close'].values.reshape(-1, 1)
     scaled_data = sclar.fit_transform(data['Close'].values.reshape(-1, 1))
     X_train = []
     y_train = []
     days = 60
     for x in range(days, len(scaled_data)):
          X_train.append(scaled_data[x-days:x, 0])
          y_train.append(scaled_data[x, 0])
     # st.text('Data preparation compleated')
     X_train, y_train = np.array(X_train), np.array(y_train)
     X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
     losses_fun = 'categorical_crossentropy'
     st.text('Training NuralNetwork.....')
     model = Sequential()
     model.add(LSTM(units=50, return_sequences=True,
               input_shape=(X_train.shape[1], 1)))
     model.add(Dropout(0.2))
     model.add(LSTM(units=50, return_sequences=True))
     model.add(Dropout(0.2))
     model.add(LSTM(units=50))
     model.add(Dropout(0.2))
     model.add(Dense(units=1))
     
     model.compile(optimizer='adam', loss=losses_fun, metrics=['accuracy'])
     # model.fit(X_train, y_train, epochs=5)
     # pred = model.predict(y_test)
     # pred
     model.fit(X_train, y_train, epochs=2)
     start_t = dt.datetime(2020, 1, 20)
     test = web.DataReader(stock, 'yahoo', start_t, end)
     act_price = test['Close'].values
     
     
     total_dataset = pd.concat((data['Close'], test['Close']), axis=0)
     model_input = total_dataset[len(total_dataset)-len(test)-days:].values
     model_input = model_input.reshape(-1, 1)
     model_input = sclar.fit_transform(model_input)
     
     X_test = []
     for x in range(days, len(model_input)):
         X_test.append(model_input[x-days:x, 0])
     #     y_train.append(model_input[x, 0])
     
     X_test = np.array(X_test)
     X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

     st.text('Training successful')
     # st.text('Making your prediction...')
     prediction = model.predict(X_test)
     prediction = sclar.inverse_transform(prediction)
     # st.text('Predicted.Ploting your prediction')
     st.subheader(f'actual price of {stock}')
     st.line_chart(act_price)
     st.subheader(f'predicted price of {stock}')
     st.line_chart(prediction)
     # st.text('Please wait printing prediction...')
     date = st.text_input('Please enter todays date in 20yy-mm-dd formate eg: 2022-2-3')
     if date:
          test = web.DataReader(stock, 'yahoo', str(date) , end)
          act_price = test['Close'].values


          total_dataset = pd.concat((data['Close'], test['Close']), axis=0)
          model_input = total_dataset[len(total_dataset)-len(test)-days:].values
          model_input = model_input.reshape(-1, 1)
          model_input = sclar.fit_transform(model_input)

          X_test = []
          for x in range(days, len(model_input)):
               X_test.append(model_input[x-days:x, 0])
#              y_train.append(model_input[x, 0])

          X_test = np.array(X_test)
          X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
          predict = model.predict(X_test)
          predict = sclar.inverse_transform(predict)
          st.text(f'Todays Close Prediction: {predict} ')
     else:
          pass

else:
     pass
