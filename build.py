#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # no warning printing

# machine learning
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import backend



class DeepNeuralNet(Sequential):

    '''
    Deep learning Wrapper with specific LSTM layer architecture for time series prediction.
    '''

    def __init__(self, input_size=60, feature_size=24, neurons=100, **kwargs):

        # -- initialize --
        super(DeepNeuralNet, self).__init__()

        # make parameters global
        self.input_size = input_size
        self.feature_size = feature_size
        self.neurons = neurons

        # scaler
        self.scale = MinMaxScaler(feature_range=(0,1))

        # Build the LSTM model (ARCHITECTURE)
        self.add(LSTM(neurons, return_sequences=True, input_shape=(input_size, 1)))
        self.add(LSTM(neurons, return_sequences=False))
        self.add(Dense(int((neurons+feature_size)/2)))
        self.add(Dense(feature_size))

        # compile the model
        self.compile(optimizer='adam', loss='mean_squared_error')
    
    def preprocess(self, Input):
        
        # -- Preprocess --
        # Input: a list or array of numbers with lenth input_size
        # refresh the data iteratively
        # reshape
        Input = np.array([[i] for i in Input])
        Input = self.scale.fit_transform(Input)
        Input = np.array( [Input[-self.input_size:,0]] )
        Input = np.reshape( Input, (Input.shape[0], Input.shape[1], 1) )
        norm = Input

        return norm

    def propagate(self, Input):

        # type validation
        if type(Input) != np.ndarray:
            Input = np.array(Input)
        if Input.shape[0] != self.input_size:
            raise TypeError(f'Input shape must be ({self.input_size},) not {Input.shape}')

        # preprocess the input
        norm = self.preprocess(Input)

        # propagate
        feature = self.predict(norm)

        # transform back
        return self.scale.inverse_transform(feature)[0]
    
    def dump(self, path):
        self.save_weights(path)