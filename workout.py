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



# compute time frame
Days = 4
dt = datetime.now()-timedelta(days=1)
min_now = str(int(int(dt.strftime("%M"))/5)*5)
if len(min_now) < 2: min_now = "0" + min_now
t_now = f'''{dt.strftime("%m-%d-%y")} {dt.strftime("%H")}:{min_now}'''
t = datetime.strptime(t_now, "%m-%d-%y %H:%M") - timedelta(days=Days)
t_then = t.strftime("%m-%d-%y %H:%M ")



highlight(f'draw the current state from arxPy from {t_then} to {t_now}...')
pairs = arx.pairs()
timeSets = {}
for pair in pairs:
    try:
        print(f'draw data for {pair} ...')
        data = arx.timeFrameData(pair, t_then, t_now)['data']
        #print(data)
        timeseries = [d[4] for d in data] # 4th element is the closing price
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


highlight('update hyper parameters ...')
School.learning_rate = p.learning_rate
School.batch_size = p.batch_size
School.validation_split = p.validation_split
School.epochs = p.epochs_workout
highlight('done.\n')


highlight('start training ...')
School.practice(x, y)
highlight('done.\n')