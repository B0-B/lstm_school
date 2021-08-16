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
    sleep(.5)
highlight('load modules ...')
from build import DeepNeuralNet
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from pathlib import Path
from client import client
import numpy as np
import json
from jsonWrapper import JSON
highlight('done.\n')



highlight('load config ...')
with open(Path('config.json').absolute()) as f:
    p = JSON(json.loads(f.read()))
highlight('done.\n')

# -- load arxpy client --
arx = client(p.host, p.port)


highlight('draw the current state from arxPy...')
pairs = arx.coins() 