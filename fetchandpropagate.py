#!/usr/bin/env python3

from build import DeepNeuralNet
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from pathlib import Path
from client import client
import json
class JSON(dict):
    #dot.notation access to dictionary attributes
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# load model
with open(Path('config.json').absolute()) as f:
    p = JSON(json.loads(f.read()))
    model = DeepNeuralNet(p.input_size, p.feature_size, p.epochs, p.batch_size, p.neurons)
    del DeepNeuralNet
    loadStatus = model.load_weights(Path(p.model_path).absolute())
    print(loadStatus)

# fetch some latest data as test data
c = client(p.host, p.port)
data = [i[4] for i in c.timeFrameData('xxrpzusd', '05-14-2021 9:00', '08-14-2021 9:00')['data']]
print("data", data)

# propagate 

prediction = model.propagate(data)
print(data)

