import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR

def prepare_data(seq, num_steps, train_size=0.85, dev_size=0.1, num_preds=1, standardize=False):
    random.seed(42)
    acc_labels_seq = np.array((seq[:-1] - seq[1:]) >= 0, dtype=np.int32)
    seq = seq[:-1]
    
    if standardize:
        scaler = StandardScaler().fit(seq[:int(len(seq) * (train_size))])
        seq = scaler.transform(seq)
    
    last_window_start = len(seq) - num_steps - num_preds + 1
    X = np.array([seq[i: i + num_steps] for i in range(last_window_start)])
    y = np.array([seq[i + num_steps:i + num_steps + num_preds, 0] for i in range(last_window_start)])
    acc_labels = np.array([acc_labels_seq[i + num_steps:i + num_steps + num_preds, 0] for i in range(last_window_start)])
    
    train_end = int(len(X) * (train_size))
    dev_end = int(len(X) * (dev_size + train_size))
    test_X, test_y, test_y_acc = X[dev_end:], y[dev_end:], acc_labels[dev_end:]
    
    X = X[:dev_end]
    y = y[:dev_end]
    acc_labels = acc_labels[:dev_end]
    
#     shuffle = list(range(X.shape[0]))
#     random.shuffle(shuffle)
#     X = X[shuffle]
#     y = y[shuffle]                         
#     acc_labels = acc_labels[shuffle]
                             
    train_X, train_y, train_y_acc = X[:train_end], y[:train_end], acc_labels[:train_end]
    dev_X, dev_y, dev_y_acc = X[train_end:], y[train_end:], acc_labels[train_end:]
    


    return train_X, train_y, dev_X, dev_y, test_X, test_y, train_y_acc, dev_y_acc, test_y_acc


def plot_results(predicted_data, true_data):
    plt.figure()
    plt.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len, baseline_data=None):
    plt.figure()
    plt.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        if i==0:
            plt.plot(padding + data, label='Prediction', color='green')
        else:
            plt.plot(padding + data, color='green')
            
    if baseline_data is not None:
        for i, data in enumerate(baseline_data):
            padding = [None for p in range(i * prediction_len)]
            if i==0:
                plt.plot(padding + data, label='Baseline', color='red')
            else:
                plt.plot(padding + data, color='red')
        
#     plt.ylim(np.min(true_data), np.max(true_data))
    plt.legend()
    plt.show()
    
def plot_results_multiple_hp_search(predicted_data, true_data, var_data, arima_data, prediction_len, save_dir):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    for i, data in enumerate(true_data):
        padding = [None for p in range(i * prediction_len)]
        if i==0:
            ax.plot(padding + data, label='True Price', color='blue')
        else:
            ax.plot(padding + data, color='blue')
            
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        if i==0:
            ax.plot(padding + data, label='LSTM', color='red')
        else:
            ax.plot(padding + data, color='red')
                
    for i, data in enumerate(var_data):
        padding = [None for p in range(i * prediction_len)]
        if i==0:
            ax.plot(padding + data, label='VAR', color='green')
        else:
            ax.plot(padding + data, color='green')
            
    for i, data in enumerate(arima_data):
        padding = [None for p in range(i * prediction_len)]
        if i==0:
            ax.plot(padding + data, label='ARIMA', color='yellow')
        else:
            ax.plot(padding + data, color='yellow')
        
#     plt.ylim(np.min(true_data), np.max(true_data))
    ax.legend()
    ax.set_xlabel('Day')
    ax.set_ylabel('Price Normalized')
    ax.set_title('Model vs Baselines, 10 day prediction')
    fig.savefig(save_dir+'10_day_Plot.png')
    
def plot_rolling(true, pred, var, arima, save_dir):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true, label='True Price')
    ax.plot(pred, label='LSTM')
    ax.plot(var, label='VAR')
    ax.plot(arima, label='ARIMA')
    ax.set_xlabel('Day')
    ax.set_ylabel('Price Normalized')
    ax.set_title('Model vs Baselines, rolling prediction on Test Set')
    ax.legend()
    fig.savefig(save_dir+'rolling.png')

def generate_hyperparams(num_sets):
    hidden_units = [50, 100, 150, 200, 250, 300]
    wlen_choices = [130, 120, 110, 100, 90, 80, 70, 60]
    batch_size_choices = [16, 32, 64, 128]
    
    hyperparams = []
    
    for i in range(num_sets):
        hyperparams_set = {}
        
        r = -np.random.uniform(2, 6)
        lr = 10**r
        
        r = -np.random.uniform(3, 7)
        decay = 10**r
        
        dropout = -np.random.uniform(0.1, 0.5)
        
        num_units = np.random.choice(hidden_units)
        
        wlen = np.random.choice(wlen_choices)
        
        batch_size = np.random.choice(batch_size_choices)
        
        hyperparams_set['lr'] = lr
        hyperparams_set['decay'] = decay
        hyperparams_set['dropout'] = dropout
        hyperparams_set['num_units'] = num_units
        hyperparams_set['wlen'] = wlen
        hyperparams_set['batch_size'] = batch_size
        
        hyperparams.append(hyperparams_set)
        
    return hyperparams

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
    
def var_predict(train_data, num_out):
    var_preds = []
    for x in train_data:
        var = VAR(x)
        var_fit = var.fit(2)
        yhat = var_fit.forecast(var_fit.y, steps=num_out)
        var_preds.append(yhat[:, 0])

    return np.array(var_preds) 

def arima_predict(train_data, num_out):
    arima_preds = []
    for x in train_data:
        arima = ARIMA(x, order=(0, 1, 1))
        arima_fit = arima.fit(disp=-1)
        yhat = arima_fit.forecast(steps=num_out)
        arima_preds.append(yhat[0])

    return np.array(arima_preds) 

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