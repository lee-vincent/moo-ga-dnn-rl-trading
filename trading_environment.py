import torch


class TradingEnvironment:
    """
    A class to simulate trading in a stock market.
    The environment is initialized with a dataset and a model.
    The model is used to make trading decisions based on the dataset.
    Profit and drawdown are calculated based on the trading decisions.
    """

    def __init__(self, features, model, opening_prices, force_cpu):
        self.features = features  # The dataset
        self.model = model  # The model
        self.opening_prices = opening_prices  # Next days' opening prices

        self.force_cpu = force_cpu

        self.initial_balance = 100_000.00  # Initial balance
        self.balance = self.initial_balance  # Balance
        self.max_balance = self.initial_balance  # Max profit
        self.drawdown = 0.00  # Drawdown
        self.shares_owned = 0  # Shares owned
        self.profit: float = 0.00  # Profit percentage
        self.num_trades = 0  # Number of trades

    def reset(self):
        """Resets the environment."""
        self.balance = self.initial_balance
        self.max_balance = self.balance
        self.drawdown = 0.00
        self.shares_owned = 0
        self.profit = 0.00
        self.num_trades = 0

    def set_model(self, new_model):
        """Sets the model."""
        self.model = new_model

    def set_features(self, new_features):
        """Sets the features."""
        self.features = new_features

    def set_opening_prices(self, new_opening_prices):
        """Sets the closing prices."""
        self.opening_prices = new_opening_prices

    def simulate_trading(self):
        """
        Currently the core of this class.
        Resets profit and drawdown for evaluation of new individual.
        Simulates trading over the dataset.
        Updates max_profit and drawdown.
        Returned values are used to evaluate the individual (multi-objective).
        No fractional shares.  Buy as many shares as possible with the money available.
        This might sound risky, but with hyper-conservative risk management (multiple stop loss variations and option protection) this is not a problem.
        Backtesting engines allow you adjust the amount invested by percentage, number of shares, etc.
        Buy and Sell signals should be executed at the next days opening price.

        """

        self.reset()
        local_decisions = []

        # Simulate trading over the dataset
        # print("self.features", self.features)
        for i in range(len(self.features)):  # this is all the rows in  training_tqqq_prepared.csv
            feature_vector = self.features[i:i+1]  # Get the feature vector for the current day
            if self.force_cpu:
                feature_vector = feature_vector.to(torch.device("cpu"))
            else:
                feature_vector = feature_vector.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            decision = self.model(feature_vector).argmax().item()  # 0=buy, 1=hold, 2=sell
            local_decisions.append(decision)
            current_price = self.opening_prices.iloc[i]  # i was adjusted in prepare_data so no need to add +1
            alpaca_finra_transaction_fee_per_share = 0.000166
            max_alpaca_finra_transaction_fee_per_trade = 8.30

            if decision == 0 and (self.balance - alpaca_finra_transaction_fee_per_share) >= current_price:  # can afford to buy at least 1 share including fee
                shares_bought = self.balance // current_price
                # figure out how many shares including fee you can afford -> we know it is at least 1
                while True:
                    # Calculate the total transaction cost
                    cost_of_shares = current_price * shares_bought
                    alpaca_finra_transaction_fee = min((alpaca_finra_transaction_fee_per_share * shares_bought), max_alpaca_finra_transaction_fee_per_trade)
                    if ((alpaca_finra_transaction_fee + cost_of_shares) <= self.balance):
                        break  # we know we can afford all those shares plus the fee
                    else:
                        shares_bought -= 1  # the fees are putting us over budget, so try buying 1 less share. we know we can afford at least 1 share including fee

                self.shares_owned += shares_bought
                self.balance -= (alpaca_finra_transaction_fee + cost_of_shares)
                self.num_trades += 1

            elif decision == 2 and self.shares_owned > 0:  # Sell
                self.balance += (current_price * self.shares_owned)
                self.balance -= (self.shares_owned * alpaca_finra_transaction_fee_per_share)  # dont think this should ever make balance negative
                self.shares_owned = 0
                self.num_trades += 1

            current_portfolio_value = self.balance + (self.shares_owned * current_price)
            self.max_balance = max(self.max_balance, current_portfolio_value)
            current_drawdown = self.max_balance - current_portfolio_value if current_portfolio_value < self.max_balance else 0.00
            drawdown_pct = (current_drawdown / self.max_balance) * 100
            self.drawdown = max(self.drawdown, drawdown_pct)

        # for the purposes of training, need to calculate portfolio value on the last day including owned shares
        # not including alpaca_finra_transaction_fee here because we just want out portfolio value
        if self.shares_owned > 0:
            self.balance += self.shares_owned * self.opening_prices.iloc[-1]
            self.shares_owned = 0
            self.num_trades += 1

        raw_profit = self.balance - self.initial_balance
        scaled_profit = raw_profit / self.initial_balance
        profit_pct = scaled_profit * 100
        self.profit = profit_pct

        return self.profit, self.drawdown
