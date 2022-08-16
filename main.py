import numpy as np
import os

import pandas as pd
from keras.layers import LSTM, Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from lemon import api
from dotenv import load_dotenv
from backtesting import Backtest, Strategy
from preprocess import get_data, get_data_from_api

load_dotenv()
client = api.create(
    market_data_api_token=os.environ["DATA_API_KEY"],
    trading_api_token=os.environ["PAPER_TRADING_API_KEY"],
    env='paper'
)


def dense_model(x_train_dense, x_test_dense, y_train_dense, y_test_dense, num_hours=60, batch_size=1, epochs=5):
    """
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
    :param model: the neural network model being used
    :param num_hours: number of hours used in each prediction
    :return: True if the predicted price is higher than the latest price, False if not
    """
    # LSTM requires [0][0], Dense does not
    prediction = model.predict(last_n_close_prices)[0][0]
    if prediction > latest_close:
        return True
    else:
        return False


def place_order(isin, num_hours, model, quantity, data_frames):
    """
    :param isin: isin of stock you wish to trade
    :param num_hours: number of hours used in each prediction
    :param model: the model being used for prediction
    :param quantity: the quantity of stock/ETF being traded each time the function is called
    :return: None
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
    last_n_close_prices = np.reshape(last_n_close_prices, (1, last_n_close_prices.shape[0]))

    if nn_trader_decision(
            num_hours=num_hours,
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


class NeuralNetTrader(Strategy):
    num_hours = 60
    x_train, y_train, x_test, y_test, test_data_frames = get_data('data.csv', num_hours=num_hours)

    # Dense Model
    # dense, y_hats_dense = dense_model(
    #     x_train_dense=x_train,
    #     x_test_dense=x_test,
    #     y_train_dense=y_train,
    #     y_test_dense=y_test,
    #     num_hours=num_hours
    # )

    # LSTM Model
    lstm, y_hats_lstm = lstm_model(
        x_train_lstm=x_train,
        x_test_lstm=x_test,
        y_train_lstm=y_train,
        y_test_lstm=y_test
    )

    def init(self):
        pass

    def next(self):
        if self.data.Close.shape[0] >= num_hours:
            latest_close_prices = np.array(self.data.Close[-1 * num_hours:], dtype=float)
            # shape for LSTM is (shape[0], 1, 1) for Dense it is (1, shape[0])
            latest_close_prices = np.reshape(latest_close_prices, (latest_close_prices.shape[0], 1, 1))
            if nn_trader_decision(
                    model=self.lstm,
                    last_n_close_prices=latest_close_prices,
                    latest_close=self.data.Close[-1]
            ):
                self.buy(size=2)
            else:
                self.sell(size=2)
        else:
            pass


if __name__ == '__main__':
    # hyper-parameters
    num_hours = 60
    trade_quantity = 2
    batch_size = 1
    epochs = 5
    isin = 'US0378331005'

    # This writes OHLC data into a csv for the selected stock/ETF (change the ISIN to change the stock/ETF).
    # Uncomment if no data.csv exists!
    # get_data_from_api(isin=isin, start_date=datetime(year=2021, month=8, day=6))

    x_train, y_train, x_test, y_test, test_data_frames = get_data('data.csv', num_hours=num_hours)

    # Dense Model
    # dense_model, y_hats_dense = dense_model(
    #     x_train_dense=x_train,
    #     x_test_dense=x_test,
    #     y_train_dense=y_train,
    #     y_test_dense=y_test
    # )
    # LSTM Model
    # lstm_model, y_hats_lstm = lstm_model(
    #     x_train_lstm=x_train,
    #     x_test_lstm=x_test,
    #     y_train_lstm=y_train,
    #     y_test_lstm=y_test
    # )
    # Change the input model(dense_model or lstm_model) to change what you base your decision on.
    # place_order(isin=isin, model=dense_model, num_hours=num_hours, quantity=trade_quantity, data_frames=test_data_frames)

    # Compare predictions of both models to the actual OHLC.
    # plt.plot(np.array(range(len(y_test), len(y_train) + len(y_test))), y_train)  # add training data plot to the graph
    # plt.plot(y_test)
    # plt.plot(y_hats_dense)
    # plt.plot(y_hats_lstm)
    # plt.legend(['Validation', 'Dense Pred.', 'LSTM Pred.'], loc='lower right')
    # plt.show()

    bt = Backtest(
        data=test_data_frames,
        strategy=NeuralNetTrader,
        cash=100000,
        commission=0,
        exclusive_orders=True
    )
    output = bt.run()
    bt.plot()
