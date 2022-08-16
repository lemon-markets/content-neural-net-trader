import csv
import math
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os
from lemon import api
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import tensorflow as tf


load_dotenv()
client = api.create(
    market_data_api_token=os.environ["DATA_API_KEY"],
    trading_api_token=os.environ["PAPER_TRADING_API_KEY"],
    env='paper'
)


def get_data_from_api(isin, start_date, end_date=datetime.now()):
    def daterange(date1, date2):
        for n in range(int((date2 - date1).days) + 1):
            yield date1 + timedelta(n)

    with open("data.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['', 'Open', 'High', 'Low', 'Close', 'Volume'])
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
                if response.pages == 0:
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
    data_set = pd.read_csv(filepath)
    data_frames = pd.DataFrame(data_set)

    close_prices = np.array(data_frames.values[:, 4])

    testing_data_len = math.ceil(len(close_prices) * 0.2)
    train_prices = close_prices[testing_data_len:]
    test_prices = close_prices[:testing_data_len]

    x_train_prices = []
    y_train_prices = []

    for i in range(num_hours, len(train_prices)):
        x_train_prices.append(train_prices[i-num_hours:i])
        y_train_prices.append(train_prices[i])

    x_train_prices, y_train_prices = np.array(x_train_prices, dtype=float), np.array(y_train_prices, dtype=float)
    x_train_prices = np.reshape(x_train_prices, (x_train_prices.shape[0], x_train_prices.shape[1], 1))

    x_test_prices = []
    y_test_prices = []

    for i in range(num_hours, len(test_prices)):
        x_test_prices.append(test_prices[i-num_hours:i])
        y_test_prices.append(test_prices[i])

    x_test_prices, y_test_prices = np.array(x_test_prices, dtype=float), np.array(y_test_prices, dtype=float)
    x_test_prices = np.reshape(x_test_prices, (x_test_prices.shape[0], x_test_prices.shape[1], 1))

    test_data_frames = data_frames.iloc[0:x_test_prices.shape[0], :]

    return x_train_prices, y_train_prices, x_test_prices, y_test_prices, test_data_frames
