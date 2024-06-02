from alpaca_api import AlpacaConnect
from assets import assets
from datetime import datetime
import math

# Get the current date and time
current_dateTime = datetime.now()

def getDay():
    """
    Get the current day as a two-digit string.
    
    Returns:
        str: Current day as 'DD'.
    """
    day = current_dateTime.day
    if day < 10:
        return '0' + str(day)
    else:
        return str(day)
    
def getMonth():
    """
    Get the current month as a two-digit string.
    
    Returns:
        str: Current month as 'MM'.
    """
    month = current_dateTime.month
    if month < 10:
        return '0' + str(month)
    else:
        return str(month)

class tradingEngine:
    """
    A class to represent a trading engine that interacts with the Alpaca API.
    
    Attributes:
        account_info (dict): Dictionary to store account and portfolio information.
        assets (dict): Dictionary of assets to manage.
        alpaca (AlpacaConnect): Instance of the AlpacaConnect class for API interaction.
        run_locally (bool): Flag to determine if the engine should run locally or not.
    """

    def __init__(self, 
                 api_key: str, 
                 secret_key: str, 
                 recommendations: dict,
                 percent_to_invest: float = None,
                 run_locally: bool = False):
        """
        Initialize the trading engine with API credentials and recommendations.
        
        Args:
            api_key (str): API key for Alpaca.
            secret_key (str): Secret key for Alpaca.
            recommendations (dict): Dictionary of stock recommendations and share prices.
            percent_to_invest (float, optional): Percentage of total cash to invest in each asset. Defaults to None.
            run_locally (bool, optional): Flag to determine if the engine should run locally. Defaults to False.
        """
        self.account_info = {}
        self.assets = assets
        if percent_to_invest is not None:
            self.assets['TQQQ']['percent_to_invest'] = percent_to_invest
        self.account_info['portfolio'] = {}
        for key in self.assets:
            self.account_info['portfolio'][key] = {}
            self.account_info['portfolio'][key]['qty'] = 0
            self.account_info['portfolio'][key]['recommendation'] = \
                recommendations[key]['recommendation']
            self.account_info['portfolio'][key]['share_price'] = \
                float(recommendations[key]['share_price'])
        
        self.alpaca = AlpacaConnect(api_key, secret_key)
        self.run_locally = run_locally
    
    def main(self):
        """
        Main function to execute the trading logic, including order placement and portfolio management.
        
        Returns:
            dict: Updated account information if not running locally.
        """
        self.alpaca.cancel_all_orders()

        account_data = self.alpaca.get_account_summary()
        self.account_info['account_value'] = account_data['portfolio_value']
        self.account_info['cash'] = account_data['cash']
        self.account_info['position_value'] = account_data['position_market_value']

        portfolio_response = self.alpaca.get_all_open_positions()

        for i in portfolio_response:
            symbol = i['symbol']
            if i.get('qty'):
                self.account_info['portfolio'][symbol]['qty'] = i.get('qty')
                self.account_info['portfolio'][symbol]['buy_price'] = float(i.get('avg_entry_price'))
            else:
                self.account_info['portfolio'][symbol]['qty'] = 0
                self.account_info['portfolio'][symbol]['buy_price'] = 0
        
        for key in self.assets:
            if self.account_info['portfolio'][key]['recommendation'] == 'buy' and int(
                float(self.account_info['portfolio'][key]['qty'])) == 0:
                    cash = float(self.account_info['cash'])
                    share_price = self.account_info['portfolio'][key]['share_price']
                    qty_to_buy = str(math.floor(cash *
                                            self.assets[key][
                                                'percent_to_invest']
                                            / float(share_price)))
                    r = self.alpaca.place_order(ticker_symbol=key,
                                            qty=qty_to_buy,
                                            side="buy",
                                            order_type="market",
                                            time_in_force="gtc")

            elif self.account_info['portfolio'][key][
                "recommendation"] == 'sell' and \
                    int(float(self.account_info['portfolio'][key]['qty'])) > 0:
                r = self.alpaca.close_position(ticker=key)

        activity_response = self.alpaca.get_activity()

        activitiy_length = len(activity_response)
        if activitiy_length >= 10:
            self.account_info['activity'] = activity_response[0:10]
        else:
            self.account_info['activity'] = activity_response[0:activitiy_length]

        orders_response = self.alpaca.get_open_orders()
        order_ids = []
        unique_orders = []
        for i in orders_response:
            if i["id"] not in order_ids:
                order_ids += [i["id"]]
                unique_orders += [i]
        self.account_info['orders'] = unique_orders

        if self.run_locally:
            # Print account summary
            print("Account Data ----------------------------------")
            print(
                f"Account Value: {float(self.account_info['account_value']):.2f} "
                f"| Cash: {float(self.account_info['cash']):.2f} "
                f"| Value of Holdings {float(self.account_info['position_value']):.2f}")
            print()
            # Print portfolio data
            print("Portfolio Data ----------------------------------")
            for i in portfolio_response:
                print(f"Symbol: {i['symbol']} "
                      f"| Quantity: {i['qty']} "
                      f"| Avg Price: {float(i['avg_entry_price']):.2f} "
                      f"| Market Value {float(i['market_value']):.2f} "
                      f"| Profit/Loss {float(i['unrealized_pl']):.2f}")
            print()
            # Print open orders
            print("Open Orders    ----------------------------------")
            for i in unique_orders:
                if i["order_type"] == 'market':
                    print(f"Symbol: {i['symbol']} "
                          f"| Quantity: {i['qty']} "
                          f"| Side: {i['side']} "
                          f"| Order Type: {i['order_type']} "
                          f"| Status: {i['status']}"
                          f"| Order ID: {i['id']}")
                elif i["order_type"] == 'trailing_stop':
                    print(f"Symbol: {i['symbol']} "
                          f"| Quantity: {i['qty']} "
                          f"| Side: {i['side']} "
                          f"| Order Type: {i['order_type']} "
                          f"| Stop Price: {i['stop_price']} "
                          f"| Status: {i['status']}"
                          f"| Order ID: {i['id']}")
                elif i["order_type"] == 'stop':
                    print(f"Symbol: {i['symbol']} "
                          f"| Quantity: {i['qty']} "
                          f"| Side: {i['side']} "
                          f"| Order Type: {i['order_type']} "
                          f"| Stop Price: {i['stop_price']} "
                          f"| Status: {i['status']} "
                          f"| Order ID: {i['id']}")
            print()
            # Print daily activity
            print("Daily Activity  ----------------------------------")
            for i in self.account_info['activity']:
                print(f"Symbol: {i['symbol']} "
                      f"| Quantity: {i['qty']} "
                      f"| Side: {i['side']} "
                      f"| Price: {i['price']} "
                      f"| Order Type: {i['type']}")
            else:
                print("No Activity today")

        else:
            # Return account info for non-local runs
            return self.account_info
