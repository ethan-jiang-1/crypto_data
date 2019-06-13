import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def prepare_data(seq, num_steps, train_size=0.8, dev_size=0.1, num_preds=1):
    last_window_start = len(seq) - num_steps - num_preds + 1
    X = np.array([seq[i: i + num_steps] for i in range(last_window_start)])
    y = np.array([seq[i + num_steps:i + num_steps + num_preds, 0] for i in range(last_window_start)])
    train_end = int(len(X) * (train_size))
    dev_end = int(len(X) * (dev_size + train_size))
    train_X, train_y = X[:train_end], y[:train_end]
    dev_X, dev_y = X[train_end:dev_end], y[train_end:dev_end]
    test_X, test_y = X[dev_end:], y[dev_end:]

    return train_X, train_y, dev_X, dev_y, test_X, test_y, y


def plot_results(predicted_data, true_data):
    plt.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    plt.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
    plt.show()

def direction_accuracy(true_data, predicted_data):
    labels = (true_data[:-1] - true_data[1:]) >= 0
    predicted = (predicted_data[:-1] - predicted_data[1:]) >= 0
    acc = np.sum(labels == predicted) / labels.size
    return acc


# def prepare_data_many_to_one(seq, num_steps, test_ratio, num_preds=1):
#     last_window_start = len(seq) - num_steps - num_preds + 1
#     X = np.array([seq[i: i + num_steps] for i in range(last_window_start)])
#     y = np.array([seq[i + num_steps:i + num_steps + num_preds, 0] for i in range(last_window_start)])
#     print(X.shape, y.shape)
#     train_size = int(len(X) * (1.0 - test_ratio))
#     train_X, test_X = X[:train_size], X[train_size:]
#     train_y, test_y = y[:train_size], y[train_size:]

#     return train_X, train_y, test_X, test_y

# def prepare_data_many_to_many(seq, num_steps, test_ratio):
#     X = np.array([seq[i: i + num_steps] for i in range(len(seq) - num_steps - 1)])
#     y = np.array([seq[i+1:i + num_steps + 1, 0] for i in range(len(seq) - num_steps - 1)])
#     train_size = int(len(X) * (1.0 - test_ratio))
#     train_X, test_X = X[:train_size], X[train_size:]
#     train_y, test_y = y[:train_size, :, [0]].squeeze(), y[train_size:, :, [0]].squeeze()

#     return train_X, train_y, test_X, test_y