#!/usr/bin/env python3

'''
This script is intended to train the AI with all available data by the current 
state of the arxPy API database. This can be triggered before the job is started, but only once.
This ensures that all the data (until now) were utilized and daily training schedule can continue.
The "stupid" model will metaphorically "catch up" on the data it has missed.
'''

from time import sleep
def highlight (stdout):
    print(f"\t\033[1;33m{stdout}\033[1;35m")
    sleep(.2)
highlight('INITIALIZE CATCHUP')    
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
model = DeepNeuralNet(p.input_size, p.feature_size, p.epochs, p.batch_size, p.neurons)
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



highlight('slice data into training sets ...')
x, y = [], []
screening_step = int(p.input_size/2)
for pair, Set in timeSets.items():
    for i in range(0, len(Set)-p.input_size-p.feature_size, screening_step):
        x.append(Set[i:i+p.input_size])
        y.append(Set[i+p.input_size:i+p.input_size+p.feature_size])
highlight('done.\n')



highlight('start training ...')
School.practice(x, y)
highlight('done.\n')