import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv
from keras.layers import LSTM, Dense
from keras.models import Sequential
from lemon import api
from pytz import utc

from preprocess import get_data, get_data_from_api

load_dotenv()
client = api.create(
    market_data_api_token=os.environ["DATA_API_KEY"],
    trading_api_token=os.environ["PAPER_TRADING_API_KEY"],
    env='paper'
)


def dense_model(x_train_dense, x_test_dense, y_train_dense, y_test_dense, num_hours=60, batch_size=1, epochs=5):
    """
    Neural network model that attempts to predict stock prices with several dense layers.

    :param epochs: number of epochs for which the model is trained - default of 5
    :param batch_size: size of one batch for training - default of 1
    :param num_hours: number of hours used to predict the next hour, default is 60
    :param x_train_dense: a np array with training inputs
    :param x_test_dense: a np array with testing inputs
    :param y_train_dense: a np array with expected training outputs
    :param y_test_dense: a np array with expected testing outputs
    :return: the trained model and its predictions on testing inputs
    """
    x_train_array = np.reshape(x_train_dense, (x_train_dense.shape[0], num_hours))
    x_test_array = np.reshape(x_test_dense, (x_test_dense.shape[0], num_hours))
    dense_model_layers = Sequential([
        Dense(50, activation='leaky_relu'),
        Dense(25, activation='leaky_relu'),
        Dense(1)
    ])
    dense_model_layers.compile(optimizer='adam', loss='mse')
    dense_model_layers.fit(x_train_array, y_train_dense, batch_size=batch_size, epochs=epochs)

    y_hats = dense_model_layers.predict(x_test_array)

    dense_rmse = np.sqrt(np.mean(y_hats - y_test_dense) ** 2)
    print(f"Dense Model RMSE: {dense_rmse}")
    return dense_model_layers, y_hats


def lstm_model(x_train_lstm, x_test_lstm, y_train_lstm, y_test_lstm, batch_size=1, epochs=5):
    """
    Neural network model that attempts to predict stock prices with LSTM and dense layers.

    :param epochs: number of epochs for which the model is trained - default of 5
    :param batch_size: size of one batch for training - default of 1
    :param x_train_lstm: a np array with training inputs
    :param x_test_lstm: a np array with testing inputs
    :param y_train_lstm: a np array with expected training outputs
    :param y_test_lstm: a np array with expected testing outputs
    :return: the trained model and the predictions it made on testing inputs
    """
    lstm_model_layers = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=(x_train_lstm.shape[1], 1)),
        LSTM(units=64),
        Dense(25, activation='leaky_relu'),
        Dense(1)
    ])
    lstm_model_layers.summary()
    lstm_model_layers.compile(optimizer='adam', loss='mse')
    lstm_model_layers.fit(x_train_lstm, y_train_lstm, batch_size=batch_size, epochs=epochs)
    y_hats = lstm_model_layers.predict(x_test_lstm)

    lstm_rmse = np.sqrt(np.mean(y_hats - y_test_lstm) ** 2)
    print(f"LSTM Model RMSE: {lstm_rmse}")

    return lstm_model_layers, y_hats


def nn_trader_decision(model, latest_close, last_n_close_prices):
    """
    Uses neural network model to make a prediction and make a decision on buy/sell after comparing to latest close.

    :param last_n_close_prices: close prices for last n hours being used for prediction
    :param latest_close: latest close price
    :param model: the neural network model being used
    :return: True if the predicted price is higher than the latest price, False if not
    """
    prediction = 0
    # LSTM requires [0][0], Dense does not due to different matrix shapes
    if model._name == 'lstm':
        prediction = model.predict(last_n_close_prices)[0][0]
    elif model._name == 'dense':
        prediction = model.predict(last_n_close_prices)
    if prediction > latest_close:
        return True
    else:
        return False


def place_order(isin, num_hours, model, quantity, data_frames):
    """
    Places buy and sell orders for the selected stock using the lemon.markets API.

    :param isin: isin of stock you wish to trade
    :param num_hours: number of hours used in each prediction
    :param model: the model being used for prediction
    :param quantity: the quantity of stock/ETF being traded each time the function is called
    """
    venue = client.market_data.venues.get(mic='xmun').results[0]
    if not venue.is_open:
        print(f"Selected venue, {venue.name}, is not open.")

    latest_close = client.market_data.ohlc.get(
        period='h1',
        isin=isin,
        mic='xmun',
        from_='latest',
        decimals=True).results[0].c

    close_prices = np.array(data_frames.values[:, 4], dtype=float)
    last_n_close_prices = close_prices[-1 * num_hours:]
    # Reshape the input to fit the model and prediction possible. When training, shape was:
    #   number of training examples x num_hours
    # So, we use 1 x num_hours here to make a single prediction.
    last_n_close_prices = np.reshape(last_n_close_prices, (1, last_n_close_prices.shape[0]))

    if nn_trader_decision(
            model=model,
            latest_close=latest_close,
            last_n_close_prices=last_n_close_prices
    ):
        price = client.market_data.quotes.get_latest(isin=isin).results[0].a
        if price * quantity < 50:  # make sure the order amount is large enough to pass through the api
            print(f"This order totals, €{price * quantity}, which is below the minimum order amount of €50.")
        # create a buy order if True is returned by MR decision function
        try:
            print('buy')
            placed_order = client.trading.orders.create(
                isin=isin,
                expires_at=7,
                side="buy",
                quantity=quantity,
                venue='xmun',
            )
            order_id = placed_order.results.id
            # subsequently activate the order
            activated_order = client.trading.orders.activate(order_id)
            print(activated_order)
        except Exception as e:
            print(f'1{e}')
    else:
        try:
            # create a sell order if mean reversion decision returns False
            print('sell')
            placed_order = client.trading.orders.create(
                isin=isin,
                expires_at=7,
                side="sell",
                quantity=quantity,
                venue='xmun',
            )
            # if position in portfolio, activate order
            if placed_order is not None:
                order_id = placed_order.results.id
                activated_order = client.trading.orders.activate(order_id)
                print(activated_order)
            else:
                print("You do not have sufficient holdings to place this order.")
        except Exception as e:
            print(f'2{e}')


if __name__ == '__main__':
    # hyper-parameters
    num_hours = 60
    trade_quantity = 2
    batch_size = 1
    epochs = 5
    isin = 'US0378331005'

    # data file
    filepath = 'data/aapl.csv'

    # Writes OHLC data into a csv for the selected stock/ETF (change the ISIN to change the stock/ETF).
    if not os.path.exists(filepath):
        get_data_from_api(filepath=filepath, isin=isin, start_date=datetime(year=2021, month=8, day=6))

    x_train, y_train, x_test, y_test, test_data_frames = get_data(filepath=filepath, num_hours=num_hours)

    dense_model, y_hats_dense = dense_model(
        x_train_dense=x_train,
        x_test_dense=x_test,
        y_train_dense=y_train,
        y_test_dense=y_test
    )
    dense_model._name = 'dense'

    lstm_model, y_hats_lstm = lstm_model(
        x_train_lstm=x_train,
        x_test_lstm=x_test,
        y_train_lstm=y_train,
        y_test_lstm=y_test
    )
    lstm_model._name = 'lstm'

    # Change the input model(dense_model or lstm_model) to change what you base your decision on.
    # place_order(isin=isin, model=dense_model, num_hours=num_hours, quantity=trade_quantity,
    #             data_frames=test_data_frames)

    scheduler = BlockingScheduler(timezone=utc)  # coordinated universal time, CET is UTC+1 (CEST is UTC+2)
    for x in range(13):
        scheduler.add_job(
            place_order, kwargs={"isin": isin,
                                 "model": dense_model,
                                 "num_hours": num_hours,
                                 "quantity": trade_quantity,
                                 "data_frames": test_data_frames},
            trigger=CronTrigger(day_of_week="mon-fri",
                                hour=6 + x,
                                minute=30,
                                timezone=utc),
            name="Perform Trades Hourly")

    print('Press Ctrl+{0} to exit'.format('Break' if os.name == 'nt' else 'C'))

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass

    # Compare predictions of both models to the actual OHLC.
    plt.plot(y_test)
    plt.plot(y_hats_dense)
    plt.plot(y_hats_lstm)
    plt.legend(['Validation', 'Dense Pred.', 'LSTM Pred.'], loc='lower right')
    plt.show()
