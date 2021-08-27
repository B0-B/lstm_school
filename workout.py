#!/usr/bin/env python3

'''
A workout training with last couple of days of data
'''

from datetime import datetime, timedelta
from time import sleep

from wrapt.wrappers import transient_function_wrapper
def highlight (stdout):
    print(f"\t\033[1;33m{stdout}\033[1;35m")
    sleep(.2)
highlight('INITIALIZE WORKOUT')    
highlight('load modules ...')
from build import DeepNeuralNet
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from pathlib import Path
from client import client
import numpy as np
import json
from jsonWrapper import JSON
from traceback import print_exc
from train import school
highlight('done.\n')



highlight('load config ...')
with open(Path('config.json').absolute()) as f:
    p = JSON(json.loads(f.read()))
highlight('done.\n')


# -- load arxpy client --
arx = client(p.host, p.port)



highlight('load model ...')
weightPath = Path(p.model_path)
highlight(f'check if {weightPath.absolute()} exists')
model = DeepNeuralNet(p.input_size, p.feature_size, p.neurons)
del DeepNeuralNet
if any(Path(weightPath.parent).iterdir()):
    highlight(f'weights found, load ...')
    loadStatus = model.load_weights(weightPath.absolute())
    print(loadStatus)
else:
    highlight(f'nothing found, initialize new model ...')
highlight('done.\n')



highlight('load school ...')
School = school(model, dumpPath=p.model_path)
highlight('done.\n')



# get all data
highlight('draw the current state from arxPy...')
pairs = arx.pairs()
timeSets = {}
for pair in pairs:
    try:
        print(f'draw data for {pair} ...')
        timeseries = [d[4] for d in arx.allData(pair)['data']] # 4th element is the closing price
        timeSets[pair] = timeseries
        print(f'success!', end='\r')
    except:
        print(f'failed.', end='\r')
        sleep(.2)
        #print_exc()
highlight('\ndone.\n')



highlight('create a random pool of training data ...')
x, y = [], []
step = int(p.input_size+p.feature_size)
for Set in timeSets.values(): # pick for each coin a random set of same size
    anc = np.random.choice(len(Set)-step-1)
    x.append(Set[anc:anc+p.input_size])
    y.append(Set[anc+p.input_size:anc+p.input_size+p.feature_size])
highlight('done.\n')



highlight('update hyper parameters ...')
School.learning_rate = p.learning_rate
School.batch_size = p.batch_size
School.validation_split = p.validation_split
School.epochs = p.epochs_workout
highlight('done.\n')



highlight('start training ...')
School.practice(x, y)
highlight('done.\n')