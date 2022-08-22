import csv
import math
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from lemon import api

load_dotenv()
client = api.create(
    market_data_api_token=os.environ["DATA_API_KEY"],
    trading_api_token=os.environ["PAPER_TRADING_API_KEY"],
    env='paper'
)


def get_data_from_api(filepath, isin, start_date, end_date=datetime.now()):
    """
    Makes request to the ohlc endpoint of the lemon.markets market data API.

    :param filepath: filepath for location/file the function will write to
    :param isin: isin of the stock data is being gathered for
    :param start_date: start date of historical period data is being gathered over
    :param end_date: end date of historical period data is being gathered over
    """
    def daterange(date1, date2):
        for n in range(int((date2 - date1).days) + 1):
            yield date1 + timedelta(n)

    with open(filepath, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
        weekdays = [6, 7]
        for dt in daterange(start_date, end_date):
            if dt.isoweekday() not in weekdays:
                response = client.market_data.ohlc.get(
                    period='h1',
                    isin=isin,
                    mic='xmun',
                    from_=dt,
                    decimals=True
                )
                if response.pages == 0:  # ignores days the market is closed and no results are found
                    continue

                for i in range(0, len(response.results)):
                    time = response.results[i].t
                    open_ = response.results[i].o
                    high = response.results[i].h
                    low = response.results[i].l
                    close = response.results[i].c
                    volume = response.results[i].v

                    writer.writerow([time, open_, high, low, close, volume])
                    print(f"{time}, {open_}, {high}, {low}, {close}, {volume}")


def get_data(filepath, num_hours):
    """
    Reads data out of csv file written by get_data_from_api function.
    Then formats data into training and testing input/output sets.

    :param filepath: filepath to the csv to read from
    :param num_hours: number of hours used to
    :return: Training and testing inputs and expected outputs.
            DataFrame of test data to be used for backtesting.
    """
    data_set = pd.read_csv(filepath)
    data_frames = pd.DataFrame(data_set)

    data_frames.index = pd.to_datetime(data_frames.values[:, 0])
    data_frames = data_frames.drop(columns='Time')

    close_prices = np.array(data_frames.values[:, 3])

    testing_data_len = math.ceil(len(close_prices) * 0.2)
    train_prices = close_prices[testing_data_len:]
    test_prices = close_prices[:testing_data_len]

    x_train_prices = []
    y_train_prices = []

    for i in range(num_hours, len(train_prices)):
        x_train_prices.append(train_prices[i-num_hours:i])
        y_train_prices.append(train_prices[i])

    x_train_prices, y_train_prices = np.array(x_train_prices, dtype=float), np.array(y_train_prices, dtype=float)
    # reshape matrices to fit the model: number of training examples x num_hours x 1
    # matrices are 3D as LSTM model in models.py requires 3D inputs
    x_train_prices = np.reshape(x_train_prices,
                                (x_train_prices.shape[0], x_train_prices.shape[1], 1))
    x_test_prices = []
    y_test_prices = []

    for i in range(num_hours, len(test_prices)):
        x_test_prices.append(test_prices[i-num_hours:i])
        y_test_prices.append(test_prices[i])

    x_test_prices, y_test_prices = np.array(x_test_prices, dtype=float), np.array(y_test_prices, dtype=float)
    # reshape matrices to fit the model: number of testing examples x num_hours x 1
    x_test_prices = np.reshape(x_test_prices,
                               (x_test_prices.shape[0], x_test_prices.shape[1], 1))

    test_data_frames = data_frames.iloc[0:x_test_prices.shape[0], :]

    return x_train_prices, y_train_prices, x_test_prices, y_test_prices, test_data_frames
