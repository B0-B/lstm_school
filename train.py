#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from traceback import print_exc
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

class school:

    def __init__(self, model, dumpPath=None):

        self.model = model
        self.x, self.y = [], [] # workload cache
        self.dumpPath = dumpPath

    def practice(self, inputs, features):

        # -- type evaluation --
        l = len(inputs)
        if l != len(features):
            raise TypeError('inputs list and features list should have same dimension.')
        
        # -- devide data into inputs and features --
        self.x, self.y = [], [] # clear cache
        for i in range(l):
            self.x.append(self.model.preprocess(inputs[i])[0])
            self.y.append(self.model.preprocess(features[i])[0]) # the 0 index accesses the only sample in list

        # -- convert to array and reshape --
        self.x, self.y = np.array(self.x), np.array(self.y)
        #self.x = np.reshape(self.x, (self.x.shape[0], self.x.shape[1], 1))

        # -- fit --
        self.history = self.model.fit(self.x, self.y, batch_size=self.model.batch_size, epochs=self.model.epochs)

        if self.dumpPath != None:
            path = Path(self.dumpPath).absolute()
            print(f'save model to {path} ...')
            self.model.dump(path)
            print('done.')
        
        return self.history.history
