#!/usr/bin/env python3

import json
from pathlib import Path
from traceback import print_exc
from datetime import datetime, timedelta
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


def waitingForSchedule (time):
    return datetime.now().strftime("%H:%M") != time
class JSON(dict):
    #dot.notation access to dictionary attributes
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


highlight('load config ...')
# load config
with open(Path('config.json').absolute()) as f:
    p = JSON(json.loads(f.read()))
highlight('done.\n')


highlight('load model ...')
weightPath = Path(p.model_path)
highlight(f'check if {weightPath.absolute()} exists')
model = DeepNeuralNet(p.input_size, p.feature_size, p.epochs, p.batch_size, p.neurons)
del DeepNeuralNet
if weightPath.exists():
    highlight(f'weights found, load ...')
    loadStatus = model.load_weights(weightPath.absolute())
    print(loadStatus)
else:
    highlight(f'nothing found, initialize new model ...')
    
#model.compile(optimizer='adam', loss='mean_squared_error')
highlight('done.\n')

# load training tool
highlight('init school ...')
gym = school(model, dumpPath=weightPath)
highlight('done.\n')

highlight('connect to arxPy API ...')
arx = client(p.host, p.port)
highlight('done.\n')

# override start minutes to match with kraken
hours = p.trigger_time.split(':')[0]
min = str(int(float(p.trigger_time.split(':')[1])/5)*5)
if len(min) < 2: min = '0' + min
startTimeOverride = f'{hours}:{min}'


if __name__ == '__main__':

    highlight(f'waiting for scheduled training at {p.trigger_time} ...')
    if p.scheduled:
        while waitingForSchedule(p.trigger_time): sleep(10)
    while True:

        try:
            
            highlight('collect datasets from arx endpoint ...')
            inputs, features = [], []
            stopDate = datetime.now().strftime('%m-%d-%y') # now
            startDate = (datetime.today() - timedelta(days=int((p.input_size+p.feature_size)/288)+1)).strftime('%m-%d-%y') # works for 5 min intervals
            for coin in arx.coins():
                try:
                    pair = coin + p.base_currency
                    dataFrame = arx.timeFrameData(pair, f'{startDate} {startTimeOverride}', f'{stopDate} {startTimeOverride}')["data"][-(p.input_size+p.feature_size):]
                    print(type(dataFrame))
                    x, y = [], []
                    for i in range((p.input_size+p.feature_size)):
                        closePrice = dataFrame[i][4]
                        if i < p.input_size:
                            x.append(closePrice)
                        else:
                            y.append(closePrice)
                    if len(x) == p.input_size and len(y) == p.feature_size:
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

            highlight(f'next training scheduled at {p.trigger_time}')

            if not p.cadence:
                break

            

        except KeyboardInterrupt:

            highlight('interrupt.')
            exit()
        
        except:

            highlight('ERROR')
            print_exc()
        
        finally:
        
            sleep(p.sleep_min*60)
