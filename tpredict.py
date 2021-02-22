%tensorflow_version 2.x
!pip install -q sklearn

import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import io
import tensorflow.compat.v2.feature_column as fc 

from tensorflow import keras
from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras 
from IPython.display import clear_output
from six.moves import urllib
from google.colab import files
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

uploaded = files.upload()
df = pd.read_csv(io.BytesIO(uploaded['TennisData.csv']))

uploaded = files.upload()
roger = pd.read_csv(io.BytesIO(uploaded['ROGER.csv']))

dataset = df.values
X = dataset[:,0:3]
Y = dataset[:,3]

min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
X_train = X_scale
Y_train = Y

dataset2 = roger.values
X2 = dataset2[:,0:3]
Y2 = dataset2[:,3]

X2_scale = min_max_scaler.fit_transform(X2)

model = Sequential([
    Dense(32,activation='relu', input_shape=(3,)),
    Dense(32,activation='relu' ),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='sgd',
              loss = 'binary_crossentropy',
              metrics=['accuracy'])
             
model.fit(X_train,Y_train,
          batch_size=32, epochs = 50)

prediction = model.predict([X2_scale])
n = 0
while (n < 7): 
  if (prediction[n] >= 0.5): 
    print('win')
  else: 
    print('loss')
  print(prediction[n])
  n = n+1
