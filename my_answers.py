import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    # Last index is `len(series) - window_size`.
    for i in range(len(series)-window_size):
        # X is extracted from i to (i + window_size) - 1.
        X.append(series[i:i+window_size])
        # y is extracted from (i + window_size).
        y.append(series[i+window_size])

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    
    # Hidden layer
    model.add(LSTM(5, input_shape=(window_size, 1)))
    # Output layer
    model.add(Dense(1))

    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = [' ', '!', ',', '.', ':', ';', '?']

    for t in text:
        # 97 is `a` and 122 is `z`.
        if t not in punctuation and (ord(t) < 97 or ord(t) > 122):
            text = text.replace(t, '')

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    # Last index is `ceil((len(series) - window_size) / step_size)`.
    for i in range(int(np.ceil((len(text)-window_size)/step_size))):
        # input is extracted from i * stepsize to (i * step_size + window_size) - 1.
        inputs.append(text[i*step_size:i*step_size+window_size])
        # output is extracted from (i * step_size + window_size)
        outputs.append(text[i*step_size+window_size])

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    
    # First layer (hidden layer).
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    # Second and Third layer (output layer).
    # Third layer is included as an activation key argument.
    model.add(Dense(num_chars, activation='softmax'))

    return model
