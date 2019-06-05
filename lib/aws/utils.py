import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import random

def prepare_data(seq, num_steps, train_size=0.8, dev_size=0.1, num_preds=1):
    random.seed(42)
    acc_labels_seq = np.array((seq[:-1] - seq[1:]) >= 0, dtype=np.int32)
    seq = seq[:-1]
    
    last_window_start = len(seq) - num_steps - num_preds + 1
    X = np.array([seq[i: i + num_steps] for i in range(last_window_start)])
    y = np.array([seq[i + num_steps:i + num_steps + num_preds, 0] for i in range(last_window_start)])
    acc_labels = np.array([acc_labels_seq[i + num_steps:i + num_steps + num_preds, 0] for i in range(last_window_start)])
    
#     shuffle = list(range(X.shape[0]))
#     random.shuffle(shuffle)
#     X = X[shuffle]
#     y = y[shuffle]                         
#     acc_labels = acc_labels[shuffle]
                             
    train_end = int(len(X) * (train_size))
    dev_end = int(len(X) * (dev_size + train_size))
    train_X, train_y, train_y_acc = X[:train_end], y[:train_end], acc_labels[:train_end]
    dev_X, dev_y, dev_y_acc = X[train_end:dev_end], y[train_end:dev_end], acc_labels[train_end:dev_end]
    test_X, test_y, test_y_acc = X[dev_end:], y[dev_end:], acc_labels[dev_end:]

    return train_X, train_y, dev_X, dev_y, test_X, test_y, train_y_acc, dev_y_acc, test_y_acc


def plot_results(predicted_data, true_data):
    plt.figure()
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
    
def plot_results_multiple_raw(predicted_data, true_data, prediction_len):
    true_data = true_data[:, 0].reshape(-1,1)
    plt.plot(true_data, label='True Data')
    
    predicted_data = predicted_data[::prediction_len].tolist()
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

def get_scores(data, preds, prediction_len):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    preds = scaler.fit_transform(preds)
    diffs = data[:-1] - data[1:]
    diffs = np.append(diffs, 0).reshape(-1,1)
    score = diffs * preds
    scores = []
    for i in range(int(len(score)/prediction_len)):
        start = i*prediction_len
        end = min((i+1)*prediction_len, len(score))
        scores.append(np.sum(score[start:end]))        
    return np.array(scores)
    


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