#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # no warning printing

# machine learning
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler



class CustomModel(Sequential):

    '''
    Deep learning Wrapper with specific LSTM layer architecture for time series prediction.
    '''

    def __init__(self, sequence_length=60, feature_length=24, epochs=1, batch_size=1, neurons=100):

        # -- initialize --
        Sequential.__init__(self)

        # make parameters global
        self.sequence_length = sequence_length
        self.feature_length = feature_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.neurons = neurons

        # scaler
        self.scale = MinMaxScaler(feature_range=(0,1))

        # Build the LSTM model (ARCHITECTURE)
        self.add(LSTM(neurons, return_sequences=True, input_shape=(sequence_length, 1)))
        self.add(LSTM(neurons, return_sequences=False))
        self.add(Dense(int((neurons+feature_length)/2)))
        self.add(Dense(feature_length))

        # compile the model
        self.compile(optimizer='adam', loss='mean_squared_error')
    
    def preprocess(self, Input):
        
        # -- Preprocess --
        # refresh the data iteratively
        Input = np.array( Input )
        Input = np.reshape( Input, (Input.shape[0], 1) )
        norm = self.scale.fit_transform(Input)

        return norm

    def propagate(self, Input):

        # type validation
        if type(Input) is not np.ndarray:
            Input = np.array(Input) # if list provided
        if Input.shape[0] != self.sequence_length:
            raise TypeError(f'Input shape must be ({self.sequence_length},) not {Input.shape}')

        norm = self.preprocess(Input)

        # propagate
        feature = self.predict(norm)

        # transform back
        return self.scale.inverse_transform(feature)