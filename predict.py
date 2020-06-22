# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 21:07:38 2019

@author: U4300
"""

from keras.layers import Dense
from keras.models import Sequential,load_model
from keras.optimizers import Adam,RMSprop
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import datetime as dt

warnings.filterwarnings('ignore')
#model = load_model('chao.h5')
filepath = r'chao.xlsx'
dataframe = pd.read_excel(filepath)
X = np.log10(dataframe.values[:,:17])
Y = np.log10(dataframe.values[:,17:])
train_x,test_x = X[:400,:],X[400:,:]
train_y,test_y = Y[:400,:],Y[400:,:]




batch_size = 30
epochs = 200

neurons = [17,10,4]
model = Sequential()
model.add(Dense(neurons[1],input_shape=(neurons[0],),activation='relu'))
model.add(Dense(neurons[1],activation='relu'))
model.add(Dense(neurons[2],activation='linear'))

model.summary()

learning_rate=0.005
algorithm = Adam(learning_rate=learning_rate)
model.compile(optimizer=algorithm,loss='mse')

callbacks = [tf.keras.callbacks.EarlyStopping(patience=2,monitor='val_loss')]
history = model.fit(train_x,train_y,batch_size=batch_size,epochs=epochs,shuffle=True,\
                    callbacks=callbacks,validation_data=(test_x,test_y))
plt.figure()
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='test')
plt.legend()
plt.show()
savepath=r'%s.h5'%dt.datetime.now().strftime('%Y%m%d%H%M%S')
model.save(savepath)

pre_Y = model.predict(X)
pre_Y = np.power(10, pre_Y)
Y = np.power(10, Y)
#X = np.power(10, X)

for i in range(len(pre_Y[0])):
    plt.figure(figsize=(15,5))
    plt.xlabel('Y')
    plt.ylabel('pre_Y')
    plt.scatter(Y[:,i],pre_Y[:,i],color='r',marker = 'o')
#    plt.scatter(pre_Y[:,i],color='blue',label='predicted')
    plt.legend()
    plt.show()

    print(mean_squared_error(Y[:,i],pre_Y[:,i]))