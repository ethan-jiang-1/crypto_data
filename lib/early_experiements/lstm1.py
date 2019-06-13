import numpy as np
import pandas as pd
import json
import os
import math
import datetime as dt
from numpy import newaxis
import tensorflow as tf
from keras.layers import Dense, Activation, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from keras.models import Sequential, load_model

from utils import *
from model_funcs import *


save_dir = 'models/lstm1'
data = pd.read_csv('data_25.csv')

num_input = 30
num_preds = 7
train_X, train_y, dev_X, dev_y, test_X, test_y = prepare_data(data, num_input, num_preds)
input_shape = train_X.shape[1:]
print("train shape", input_shape)

if not os.path.exists(save_dir): os.makedirs(save_dir)

lr = 0.001
decay = 1e-6
dropout_rate = 0.8
epochs = 200
batch_size = 32

model = Sequential()
model.add(
    CuDNNLSTM(
        128,
        input_shape=(x_train.shape[1:]),
        return_sequences=True,
    )
)
model.add(Dropout(dropout_rate))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, return_sequences=True))
model.add(Dropout(dropout_rate))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128))
model.add(Dropout(dropout_rate))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))
model.add(Dropout(dropout_rate))

model.add(Dense(num_preds, activation="softmax"))


opt = tf.keras.optimizers.Adam(lr=lr, decay=decay)
model.compile(loss="mse", optimizer=opt, metrics=["mse"])


history = train(
    model,
    train_X,
    train_y,
    (dev_X, dev_y),
    epochs = epochs,
    batch_size = batch_size,
    save_dir = save_dir
)