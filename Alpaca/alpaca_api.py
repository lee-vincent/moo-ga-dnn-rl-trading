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

            self._orders_url = f'{self.base_url}/orders'
            self._positions_url = f'{self.base_url}/positions'
            self._account_url = f'{self.base_url}/account'
            self._portfolio_url = f'{self.base_url}/account/portfolio/history'
            # Has activity url changed? TODO Research
            # self.activity_url = f'{self.base_url}/account/activities?activity=trades'


# Used in the Market Data API, needed for project?
class CurrentDate:
    # IN PROGRESS
    def __init__(self):
        self._current_datetime = datetime.now()
        self._base_url = "https://data.alpaca.markets/v2/stocks/"

    def get_day(self):
        day = self._current_datetime.day
        return f'{day:02}'

    def get_month(self):
        month = self._current_datetime.month
        return f'{month:02}'


# Used in the Market Data API, needed for project?
class BuildLongURLs:
    # IN PROGRESS
    def __init__(self):
        self._base_url = "https://data.alpaca.markets/v2/stocks/"

        self._current_datetime = CurrentDate()
        self._year = str(self._current_datetime.year)
        self._month = self._current_datetime.get_month()
        self._day = self._current_datetime.get_day()





