import requests
from datetime import datetime


class AlpacaConnect:

    def __init__(self, alpaca_key, secret_alpaca_key, paper=True):
        """
        Creates AlpacaConnect object. Defaults parameter is set to True to
        indicate paper trading. Set False to trade with real money.
        """
        self.HEADERS = {
            "accept": "application/json",
            "content-type": "application/json",
            "APCA-API-KEY-ID": alpaca_key,
            "APCA-API-SECRET-KEY": secret_alpaca_key
        }

        if paper:
            self._base_url = "https://paper-api.alpaca.markets/v2"

            self._orders_url = f'{self._base_url}/orders'
            self._positions_url = f'{self._base_url}/positions'
            self._account_url = f'{self._base_url}/account'
            self._portfolio_url = f'{self._base_url}/account/portfolio/history'
            self.activity_url = f'{self._base_url}/account/activities?activity=trades'
        # else:
            # block for real money trading not implemented yet

    def get_account_summary(self):
        """
        Get account summary
        """
        response = requests.get(self._account_url, headers=self.HEADERS)
        return response.json()

    def get_activity(self):
        """
        Get account trade activity
        """
        response = requests.get(self.activity_url, headers=self.HEADERS)
        return response.json()

    def get_portfolio_history(self):
        """
        Get portfolio history/transactions done within the last month
        """
        long_url = LongURLs()
        portfolio_url = long_url.build_portfolio_history(self._portfolio_url)
        response = requests.get(portfolio_url, headers=self.HEADERS)
        return response.json()

    def get_historical_data(self, ticker="TQQQ"):
        """
        Get the historical data. Default data starts on 07/12/2018. Returns the
        Open, High, Low, and Close of each day
        """
        long_url = LongURLs()
        historical_url = long_url.build_historical_data(ticker)
        response = requests.get(historical_url, headers=self.HEADERS).json()

        formatted_response = [["Date", "Open", "High", "Low", "Close"]]

        for data in response["bars"]:
            date = data['t'].split("T")
            formatted_response += [[date[0], data['o'], data['h'], data['l'], data['c']]]

        return formatted_response

    def get_open_orders(self):
        """
        Get all open orders
        """
        response = requests.get(f'{self._orders_url}?status=open', headers=self.HEADERS)
        return response.json()

    def get_all_orders(self):
        """
        Get all open, cancelled, and filled orders
        """
        response = requests.get(f'{self._orders_url}?status=all', headers=self.HEADERS)
        return response.json()

    def get_all_open_positions(self):
        response = requests.get(self._positions_url, headers=self.HEADERS)
        return response.json()

    def close_position(self, ticker="TQQQ"):
        """
        Closes a position of a given ticker
        """
        response = requests.delete(f'{self._positions_url}/{ticker}', headers=self.HEADERS)
        return response.json()

    def place_order(self, ticker_symbol, qty, side, order_type='market', time_in_force='gtc', stop_price=float(0)):
        """
        Places a buy or sell order with the given information.
        """
        # Create base payload with common fields
        payload = {
            'symbol': ticker_symbol,
            'qty': qty,
            'side': side,                       # buy/sell
            'type': order_type,
            'time_in_force': time_in_force,     # default gtc, good until canceled
        }

        # Add/update fields based on the order type
        if order_type == 'stop':
            payload['stop_price'] = stop_price
        elif order_type == 'limit':
            payload['limit_price'] = stop_price

        response = requests.post(f'{self._orders_url}', json=payload, headers=self.HEADERS)
        return response.json()

    def cancel_one_order(self, order_id):
        """
        Cancels an order based on the given id.
        Returns a 204 No Content response on success, otherwise
        returns a 422 Unprocessable Entity (not cancelable)
        """
        response = requests.delete(f'{self._orders_url}/{order_id}', headers=self.HEADERS)
        return response.status_code

    def cancel_all_orders(self):
        """
        Cancels all open orders
        """
        response = requests.delete(self._orders_url, headers=self.HEADERS)
        return response.json()


class CurrentDate:

    def __init__(self):
        """
        Creates simple CurrentDate object using datetime
        """
        self._current_datetime = datetime.now()

    def get_day(self):
        """
        Get current day represented in two-digit format
        """
        day = self._current_datetime.day
        return f'{day:02}'

    def get_month(self):
        """
        Get current month represented in two-digit format
        """
        month = self._current_datetime.month
        return f'{month:02}'


class LongURLs:

    def __init__(self):
        """
        Create LongURLs object used to build long urls for historical data
        and portfolio history urls
        """
        self._base_url = "https://data.alpaca.markets/v2/stocks/"
        self._current_datetime = CurrentDate()
        self._year = str(self._current_datetime.year)
        self._month = self._current_datetime.get_month()
        self._day = self._current_datetime.get_day()

    def build_historical_data(self, ticker):
        """
        Builds the historical data url of the given ticker
        """
        historical_url = (
            f'{self._base_url}{ticker}/bars?timeframe=1Day&start=2011-01-02&end='
            f'{self._year}-{self._month}-{self._day}T04%3A00%3A00Z&limit=2000&adjustment=split'
        )
        return historical_url

    def build_portfolio_history(self, base_url):
        """
        Builds the portfolio history url
        """
        portfolio_url = (
            f'{base_url}?period=1M&timeframe=1D&date_end='
            f'{self._year}-{self._month}-{self._day}'
        )
        return portfolio_url
