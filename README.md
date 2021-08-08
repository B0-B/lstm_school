# LSTM School
A training scheduleder for tensorflow LSTM models. Trains AI daily using API data and keeps model updated and prestine.

## Data
To ensure a continuous data stream for training this project utilizes [arxPy]("https://github.com/B0-B/arxPy-Crypto-Data-Miner") to aggregate data and serve API endpoint which can run on the same machine or a remote server.
Assume arxPy running on an arbitrary HOST and PORT, these two parameters need to be transfered into the client-side `config.json`
```bash
{
    "host": "http://localhost",
    "port": 8080,
    ...
}
```
this will wire school to the data endpoint.

## What can LSTM school for you?
It maintains your tensorflow Neural Network Model on any host or server. The project is coded in a transparent and toy-ish way to make customizations easy. By default a LSTM architecture is loaded (blueprint in `build.py`).
Use `config.json` to set the training parameters.
```bash
    ...
    "input_size": 600,
    "feature_size": 36,
    "neurons": 500,
    "cadence": false, 
    "model_path": "./model/model.h5",
    "batch_size": 5,
    "epochs": 2,
    "trigger_time": "11:01",
    "sleep_min": 360,
    "base_currency": "USD"
}
```
By the default settings the model will be trained every 6 hours with 600 timeseries values/prices with 5 minute increments corresponding to latest 2.08 days, with the aim to predict the next 36 values i.e. 3 hours.

## Job

The complete orchestration and workload is denoted in `job.py`.
```bash
~/lstm_school$ python job.py  
```
the latest model which can be loaded is accessible under <strong>lstm_school/model/</strong>.