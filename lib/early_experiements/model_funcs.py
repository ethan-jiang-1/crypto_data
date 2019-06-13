import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

def train(model, x, y, validation_data, epochs, batch_size, save_dir):
    print('[Model] Training Started')
    print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
    
    save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
    callbacks = [
        # EarlyStopping(monitor='val_loss', patience=7),
        ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True),
        TensorBoard(log_dir=os.path.join(save_dir, 'logs'))
    ]
    history = model.fit(
        x,
        y,
        epochs=epochs,
        validation_data=validation_data,
        batch_size=batch_size,
        callbacks=callbacks
    )
    model.save(save_fname)

    print('[Model] Training Completed. Model saved as %s' % save_fname)

    return history

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    print('[Model] Predicting Point-by-Point...')
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    print('[Model] Predicting Sequences Multiple...')
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs

def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    print('[Model] Predicting Sequences Full...')
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
    return predicted