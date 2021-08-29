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
import json
import urllib
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import Request

def OHLC (pair, intervalInMinutes, epochInMinutes):
            
    '''
    provide epoch parameter as unix timestamp
    exp.:   OHLC("XRPEUR", 5, 1440) for ripple timeseries data on 5 min interval
            of the last 24 hours. This yields 1440/5=288 data points <=> returned list length.
    '''

    URL = 'https://api.kraken.com/0/public/OHLC'

    # -- validate input --
    if type(pair) is not str or len(pair) == 0:
        raise TypeError(f'pair must be of type str e.g. "XRPEUR" not type {type(intervalInMinutes)}')
    if type(intervalInMinutes) is not int:
        raise TypeError(f'intervalInMinutes must be of type int not {type(intervalInMinutes)}')
    elif type(epochInMinutes) is not int:
        raise TypeError(f'epochInMinutes must be of type int not {type(epochInMinutes)}')
        
    # -- build data package and encode --
    # convert epoch into unix timestamp
    now = datetime.now()
    past = now - timedelta(minutes=epochInMinutes)
    data = {
        'pair' : pair,
        'interval' : intervalInMinutes,
        'since' : past.timestamp()
    }
    postdata = urlencode(data)
    body = postdata.encode("utf-8")

    # -- API request --
    try:
        request = Request(URL, data=body)
    except HTTPError:
        raise 'cannot establish connection to kraken API, but server is up'
    except Exception as e:
        raise e

    # -- validate response --
    ticket = urllib.request.urlopen(request)
    raw_data = ticket.read()
    encoding = ticket.info().get_content_charset('utf8') # JSON default
    errors = json.loads(raw_data.decode(encoding))['error']  # return ticket result
    result = json.loads(raw_data.decode(encoding))['result']
    serverTimestamp = result['last'] # get server time to sync
    serverTime = datetime.fromtimestamp(serverTimestamp)
    timeseries = result[list(result.keys())[0]]
    past = serverTime - timedelta(minutes=epochInMinutes)
    if 'EService:Busy' in errors:
        raise 'kraken API service is busy'
    
    # -- build package --
    package = {
        'data': [],
        'length': len(timeseries),
        'pair': pair,
        'start': past.strftime("%m-%d-%y %H:%M"),
        'startStamp': past.timestamp(),
        'stop': serverTime.strftime("%m-%d-%y %H:%M"),
        'stopStamp': serverTimestamp,
    }
    for value in timeseries:
        dataPoint = [ # time/o/h/l/c/avg/volume
            datetime.fromtimestamp(value[0]).strftime("%m-%d-%y %H:%M"),
            value[1],
            value[2],
            value[3],
            value[4],
            value[5],
            value[6]
        ] 
        package['data'].append(dataPoint)

    return package

COIN = "xbt"

data = OHLC(COIN+'USD', 5, 3500)
close = [float(d[4]) for d in data['data']]
plt.plot(close)
plt.show()