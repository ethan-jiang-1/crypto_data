"""
An deep RNN model for predictions on price sequence data
"""
import os
import re
from pdb import set_trace as bp
import fnmatch
from itertools import zip_longest
from collections import deque
import random
import numpy as np
import time
import pandas as pd

from pykalman import KalmanFilter
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.regularizers import L1L2
from sklearn import preprocessing
from matplotlib import pyplot


class PriceRNN:
    def __init__(
        self,
        pair="BTCUSD",
        period="1min",
        window_len=60,
        forecast_len=3,
        years=["2015", "2016", "2017", "2018", "2019"],
        epochs=10,
        dropout=0.2,
        testpct=0.15,
        loss_func="mse",
        batch_size=64,
        hidden_node_sizes=[128] * 4,
        learning_rate=0.001,
        decay=1e-6,
        scaler=preprocessing.MinMaxScaler(feature_range=(0, 1)),
        data_provider="gemini",
        data_dir="data",
        skiprows=3,
        chunksize=10_000,
    ):
        self.data_provider = data_provider
        self.data_dir = data_dir
        self.pair = pair
        self.period = period
        self.file_filter = f"{data_provider}_{pair}_*{period}.csv"
        self.window_len = window_len  # price data window
        self.forecast_len = forecast_len  # how many data points in future to predict
        self.years = years
        self.epochs = epochs
        self.dropout = dropout
        self.testpct = testpct
        self.loss_func = loss_func
        self.batch_size = batch_size
        self.hidden_node_sizes = hidden_node_sizes
        self.learning_rate = learning_rate
        self.decay = decay
        self.scaler = scaler
        self.name = f"{pair}-WLEN{window_len}-FLEN{forecast_len}-{int(time.time())}"
        self.skiprows = skiprows
        self.chunksize = chunksize
        self.col_names = ["date", "close", "open", "high", "low", "volume", "change"]
        self.file_filter = f"{data_provider}_{pair}_*{period}.csv"

    def extract_data(self):
        df = pd.DataFrame()

        df = pd.read_csv(
            f"{self.data_dir}/btc_training.csv",
            index_col="Date",
            parse_dates=True,
            usecols=["Date", "Price", "Vol2"],
        )

        df = df[(df.index >= "2017-01-01")]

        # the features we care about
        df = df[["Price", "Vol2"]]

        df.fillna(method="ffill", inplace=True)
        df.dropna(inplace=True)
        return df

    def transform_data(self, main_df):
        # add a future price target column shifted in relation to close
        main_df["target"] = main_df["Price"].shift(-self.forecast_len)
        main_df = main_df[~main_df.isin([np.nan, np.inf, -np.inf]).any(1)]

        train_df, test_df = self.split_dataset(main_df)

        # normalize
        for col in train_df.columns:
            train_df[col] = self.scaler.fit_transform(
                train_df[col].values.reshape(-1, 1)
            )

        for col in test_df.columns:
            test_df[col] = self.scaler.transform(test_df[col].values.reshape(-1, 1))

        return train_df, test_df

    # arrange, split
    def load(self, df):
        seq_data = self.convert_to_seq(df)
        # seq_data = self.balance(seq_data)

        print("SPLITTING DATA:\n", seq_data[0][0][0:1][0])
        # split data into train, test sets
        # to prevent skewing of distribution fed into network, randomize
        random.shuffle(seq_data)
        x, y = [], []
        for window_seq, target in seq_data:
            x.append(window_seq)
            y.append(target)

        return np.array(x), y

    # convert data into seq -> target pairs
    def convert_to_seq(self, df):
        print("ARRANGING DATA:\n", df.head(10))
        seq_data = []
        # acts as sliding window - old values drop off
        prev_days = deque(maxlen=self.window_len)
        for i in df.values:
            prev_days.append([n for n in i[:-1]])  # exclude target (i[:-1])
            if len(prev_days) == self.window_len:
                seq_data.append([np.array(prev_days), i[-1]])

        random.shuffle(seq_data)  # prevent skew in distribution
        return seq_data

    def split_dataset(self, main_df):
        times = sorted(main_df.index.values)
        test_cutoff = times[-int(self.testpct * len(times))]
        print("Test cutoff: ", test_cutoff)
        # SPLIT DATA INTO (test_cutoff)% TEST, (1-test_cutoff)% TRAIN
        test_df = main_df[(main_df.index >= test_cutoff)]
        train_df = main_df[(main_df.index < test_cutoff)]
        return train_df, test_df

    def run(self):
        random.seed(230)  # determinism

        # Extract, Transform, Load
        main_df = self.extract_data()
        train_df, test_df = self.transform_data(main_df)

        x_train, y_train = self.load(train_df)
        x_test, y_test = self.load(test_df)

        # shows balance
        print(
            f"x_train: {x_train.shape}, {len(x_train)}, x_test: {x_test.shape}, {len(x_test)}"
        )
        print(f"x_train.shape[1:]: {x_train.shape[1:]}")
        print(f"y_test: {len(y_train)}, y_test: {len(y_test)}")
        model = self.model(x_train)

        opt = tf.keras.optimizers.Adam(lr=self.learning_rate, decay=self.decay)

        model.compile(loss=self.loss_func, optimizer=opt, metrics=["mse", "acc"])

        if not os.path.exists("logs"):
            os.makedirs("logs")
        tensorboard = TensorBoard(log_dir=f"logs/{self.name}")

        # if not os.path.exists("models"):
        #     os.makedirs("models")
        # unique filename to include epoch and validation accuracy for that epoch
        # filepath = "RNN_Final-{epoch:02d}-{val_loss:.3f}"
        # checkpoint = ModelCheckpoint(
        #     "models/{}.model".format(
        #         filepath, monitor="val_loss", verbose=1, save_best_only=True, mode="max"
        #     )
        # )

        history = model.fit(
            x_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(x_test, y_test),
            callbacks=[tensorboard],
        )

        if not os.path.exists("plots"):
            os.makedirs("plots")
        pyplot.plot(history.history["loss"])
        pyplot.plot(history.history["val_loss"])
        pyplot.title("model train vs validation loss")
        pyplot.ylabel("loss")
        pyplot.xlabel("epoch")
        pyplot.legend(["train", "validation"], loc="upper right")
        pyplot.savefig(f"plots/{self.name}.png")

        # print(history.history["loss"])
        # print(history.history["acc"])
        # print(history.history["val_loss"])
        # print(history.history["val_acc"])
        print("Eval Metrics ", model.metrics_names)
        print(model.evaluate(x_test, y_test))
        print(model.summary())

    def model(self, x_train):
        model = Sequential()
        model.add(
            CuDNNLSTM(
                self.hidden_node_sizes[0],
                input_shape=(x_train.shape[1:]),
                return_sequences=True,
            )
        )
        model.add(Dropout(self.dropout))
        model.add(BatchNormalization())

        model.add(CuDNNLSTM(self.hidden_node_sizes[1], return_sequences=True))
        model.add(Dropout(self.dropout))
        model.add(BatchNormalization())

        model.add(CuDNNLSTM(self.hidden_node_sizes[2]))
        model.add(Dropout(self.dropout))
        model.add(BatchNormalization())

        model.add(Dense(32, activation="relu"))
        model.add(Dropout(self.dropout))

        model.add(Dense(1, activation="linear"))
        return model


# Model to answer: If you were to buy at random based on model prediction, what
# hold period shows highest probability of profit
# TODO: random search and/or bayesian hyperparam optimization
w = 120
for wlen, flen in [
    # (w, 6),
    # (w, 7),
    # (w, 8),
    # (w, 9),
    # (w, 11),
    # (w, 12),
    (w, 13),
    # (w, 14),
    # (w, 15),
    # (w, 16),
    # (w, 17),
    # (w, 18),
    # (w, 19),
    # (w, 20),
]:
    wlen = int(wlen)
    flen = int(flen)
    print("RUNNING MODEL: ")
    print("\twindow length: ", wlen)
    print("\tforecast length: ", flen)
    PriceRNN(
        pair="BTCUSD",
        period="1d",
        window_len=wlen,
        forecast_len=flen,
        dropout=0.5,
        epochs=100,
        batch_size=32,
        hidden_node_sizes=[128] * 4,
        testpct=0.25,
        learning_rate=0.001,
        decay=1e-6,
        data_dir="data",
    ).run()
