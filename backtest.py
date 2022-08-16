import numpy as np
from backtesting import Backtest, Strategy

from models import nn_trader_decision, dense_model, lstm_model
from preprocess import get_data


class DenseNeuralNetTrader(Strategy):
    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)
        self.num_hours = None
        self.lstm = None
        self.dense = None
        self.y_hats_lstm = None
        self.y_hats_dense = None

    def init(self):
        self.num_hours = 60
        x_train, y_train, x_test, y_test, test_data_frames = get_data('data/aapl.csv', num_hours=num_hours)

        # Dense Model
        self.dense, self.y_hats_dense = dense_model(
            x_train_dense=x_train,
            x_test_dense=x_test,
            y_train_dense=y_train,
            y_test_dense=y_test,
            num_hours=num_hours
        )
        self.dense._name = 'dense'

    def next(self):
        if self.data.Close.shape[0] >= self.num_hours:
            latest_close_prices = np.array(self.data.Close[-1 * self.num_hours:], dtype=float)
            latest_close_prices = np.reshape(latest_close_prices, (1, latest_close_prices.shape[0]))
            if nn_trader_decision(
                    model=self.dense,
                    last_n_close_prices=latest_close_prices,
                    latest_close=self.data.Close[-1]
            ):
                self.buy(size=2)
            else:
                self.sell(size=2)
        else:
            pass


class LSTMNeuralNetTrader(Strategy):

    def __init__(self, broker, data, params):
        super().__init__(broker, data, params)
        self.num_hours = None
        self.lstm = None
        self.dense = None
        self.y_hats_lstm = None
        self.y_hats_dense = None

    def init(self):
        self.num_hours = 60
        x_train, y_train, x_test, y_test, test_data_frames = get_data('data/aapl.csv', num_hours=num_hours)

        # LSTM Model
        self.lstm, self.y_hats_lstm = lstm_model(
            x_train_lstm=x_train,
            x_test_lstm=x_test,
            y_train_lstm=y_train,
            y_test_lstm=y_test
        )
        self.lstm._name = 'lstm'

    def next(self):
        if self.data.Close.shape[0] >= self.num_hours:
            latest_close_prices = np.array(self.data.Close[-1 * self.num_hours:], dtype=float)
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
    num_hours = 60

    x_train, y_train, x_test, y_test, test_data_frames = get_data('data/aapl.csv', num_hours=num_hours)

    bt = Backtest(
        data=test_data_frames,
        strategy=DenseNeuralNetTrader,
        cash=100000,
        commission=0,
        exclusive_orders=True
    )
    dense_output = bt.run()
    bt.plot()  # generate an informative HTML graphic for your backtest

    bt = Backtest(
        data=test_data_frames,
        strategy=LSTMNeuralNetTrader,
        cash=100000,
        commission=0,
        exclusive_orders=True
    )
    lstm_output = bt.run()
    bt.plot()
