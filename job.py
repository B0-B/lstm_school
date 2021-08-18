#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # no warning printing
import json
from tensorflow.python.keras.backend import update
from jsonWrapper import JSON
from pathlib import Path
from traceback import print_exc
from datetime import date, datetime, timedelta
from time import sleep
def highlight (stdout):
    print(f"\t\033[1;33m{stdout}\033[1;35m")
    sleep(.5)
highlight('load modules ...')
from client import client
from build import DeepNeuralNet
from tensorflow.keras.models import load_model
from train import school
highlight('done.\n')

# -- scheduler --
def waitingForSchedule (times):
    # times is an array of time format strings
    for t in times:
        if datetime.now().strftime("%H:%M") == t:
            return False
    return True


highlight('load config ...')
with open(Path('config.json').absolute()) as f:
    p = JSON(json.loads(f.read()))
highlight('done.\n')


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
    highlight(f'nothing found, run catchup script ...')
    os.system('python3 catchup.py')
highlight('done.\n')

# load training tool
highlight('init school ...')
gym = school(model, dumpPath=weightPath)
highlight('done.\n')

highlight('connect to arxPy API ...')
arx = client(p.host, p.port)
highlight('done.\n')



# compute the start time
if p.scheduled:
    if len(p.trigger_times) == 1:
        next_schedule = p.trigger_times[0]
    else:
        p.trigger_times.sort()
        t_mem = datetime.now()
        l = len(p.trigger_times)
        for i in range(l):
            if t_mem > datetime.strptime(p.trigger_times[i], "%H:%M") and t_mem < datetime.strptime(p.trigger_times[(i+1)%l], "%H:%M"):
                next_schedule = p.trigger_times[(i+1)%l]
else:
    
    next_schedule = (datetime.now() + timedelta(0, 10)).strftime("%H:%M")
    highlight(f'auto scheduled training at {next_schedule}')

if __name__ == '__main__':

    highlight(f'waiting for scheduled training at {next_schedule} ...')
    
    while True:

        
        
        if p.scheduled:
            while waitingForSchedule(next_schedule): sleep(10)
        t_mem = datetime.now().strftime("%H:%M") # memorize time

        highlight('reload config ...')
        with open(Path('config.json').absolute()) as f:
            p = JSON(json.loads(f.read()))
        highlight('done.\n')

        highlight('update hyper parameters ...')
        gym.learning_rate = p.learning_rate
        gym.batch_size = p.batch_size
        gym.validation_split = p.validation_split
        gym.epochs = p.epochs
        highlight('done.\n')

        try:
            
            highlight('collect datasets from arx endpoint ...')
            inputs, features = [], []
            stopDate = datetime.now().strftime('%m-%d-%y') # now
            startDate = (datetime.today() - timedelta(days=int((model.input_size+model.feature_size)/288)+1)).strftime('%m-%d-%y') # works for 5 min intervals
            for coin in arx.coins():
                print(f'draw data for {coin} ...')
                try:

                    # override start minutes to match with kraken
                    hours = next_schedule.split(':')[0]
                    min = str(int(float(next_schedule.split(':')[1])/5)*5)
                    if len(min) < 2: min = '0' + min
                    startTimeOverride = f'{hours}:{min}'
                    
                    # request data from arxpy
                    pair = coin + p.base_currency
                    dataFrame = arx.timeFrameData(pair, f'{startDate} {startTimeOverride}', f'{stopDate} {startTimeOverride}')["data"][-(model.input_size+model.feature_size):]
                    
                    # create one input data set and append to input and features array
                    x, y = [], []
                    for i in range((model.input_size+model.feature_size)):
                        closePrice = dataFrame[i][4]
                        if i < model.input_size:
                            x.append(closePrice)
                        else:
                            y.append(closePrice)
                    if len(x) == model.input_size and len(y) == model.feature_size:
                        inputs.append(x)
                        features.append(y)

                except Exception as e:

                    if "NoneType" in str(e) or "range" in str(e):

                        pass

                    else:

                        print_exc()

            highlight(f'{len(inputs)} new datasets collected for training.\n\tdone.\n')

            # -- backprop --
            highlight('initialize training ...')
            metrics = gym.practice(inputs, features)
            highlight('training completed.\n')
            highlight(f'loss: {100*metrics["loss"][-1]}%')

            # schedule next training only when on cadence
            if not p.cadence:
                break
            else:
                next_schedule = p.trigger_times[(p.trigger_times.index(t_mem)+1)%len(p.trigger_times)]
                highlight(f'next training scheduled at {next_schedule}\n')

        except KeyboardInterrupt:

            highlight('interrupt.')
            exit()
        
        except:

            highlight('ERROR')
            print_exc()
        
        finally:
        
            sleep(100)
