import alpaca_api
import config

api_key = config.ALPACA_KEY
secret_key = config.ALPACA_SECRET_KEY

def test_get_account_summary():
    """
    Test to retrieve and print the account summary, including account number,
    cash balance, and buying power.
    """
    test_obj = alpaca_api.AlpacaConnect(api_key, secret_key)
    result = test_obj.get_account_summary()

    print(
        f'Account Number: {result["account_number"]} '
        f'| Cash: {result["cash"]} '
        f'| Buying Power: {result["buying_power"]}'
    )

def print_historical_data(data):
    """
    Helper function to print historical data in a readable format.

    Args:
    data (list): List of historical data entries.
    """
    for i in data:
        print(f'{i[0]}, {i[1]}, {i[2]}, {i[3]}, {i[4]}')

def test_history():
    """
    Test to retrieve and print all open orders, including symbol, side, quantity,
    order type, status, and order ID.
    """
    test_obj = alpaca_api.AlpacaConnect(api_key, secret_key)
    result = test_obj.get_open_orders()
    
    for i in result:
        print(
            f'Symbol: {i["symbol"]} '
            f'| Side {i["side"]} '
            f'| Qty: {i["qty"]} '
            f'| Order Type: {i["type"]} '
            f'| Status: {i["status"]} '
            f'| Order Id: {i["id"]}'
        )

def test_place_order(ticker='TQQQ'):
    """
    Test to place a market buy order and print the details of the placed order.

    Args:
    ticker (str): Ticker symbol for the order (default: 'TQQQ').
    """
    qty = 25
    side = 'buy'
    type = 'market'
    time_in_force = 'gtc'

    test_obj = alpaca_api.AlpacaConnect(api_key, secret_key)
    result = test_obj.place_order(ticker, qty, side, type, time_in_force)

    print(
        f'Symbol: {result["symbol"]} '
        f'| Side {result["side"]} '
        f'| Qty: {result["qty"]} '
        f'| Order Type: {result["type"]} '
        f'| Status: {result["status"]} '
        f'| Order Id: {result["id"]}'
    )

def test_get_orders():
    """
    Test to retrieve and print all open orders, including symbol, side, quantity,
    order type, status, and order ID.
    """
    test_obj = alpaca_api.AlpacaConnect(api_key, secret_key)
    result = test_obj.get_open_orders()

    for i in result:
        print(
            f'Symbol: {i["symbol"]} '
            f'| Side {i["side"]} '
            f'| Qty: {i["qty"]} '
            f'| Order Type: {i["type"]} '
            f'| Status: {i["status"]} '
            f'| Order Id: {i["id"]}'
        )

def test_get_all_open_positions():
    """
    Test to retrieve and print all open positions, including symbol, quantity,
    average entry price, current price, and unrealized profit/loss.
    """
    test_obj = alpaca_api.AlpacaConnect(api_key, secret_key)
    result = test_obj.get_all_open_positions()

    for i in result:
        print(
            f'Symbol: {i["symbol"]} '
            f'| Qty: {i["qty"]} '
            f'| Avg Price: {i["avg_entry_price"]} '
            f'| Current Price: {i["current_price"]} '
            f'| Unrealized P/L: {i["unrealized_pl"]}'
        )

def test_stop_loss_order(ticker='TQQQ'):
    """
    Test to place a stop sell order and print the details of the placed order.

    Args:
    ticker (str): Ticker symbol for the order (default: 'TQQQ').
    """
    qty = 15
    side = 'sell'
    type = 'stop'
    time_in_force = 'gtc'
    stop_price = 35

    test_obj = alpaca_api.AlpacaConnect(api_key, secret_key)
    result = test_obj.place_order(ticker, qty, side, type, time_in_force, stop_price)

    print(
        f'Symbol: {result["symbol"]} '
        f'| Side {result["side"]} '
        f'| Qty: {result["qty"]} '
        f'| Order Type: {result["type"]} '
        f'| Status: {result["status"]} '
        f'| Order Id: {result["id"]}'
    )

def test_limit_order(ticker='TQQQ'):
    """
    Test to place a limit buy order and print the details of the placed order.

    Args:
    ticker (str): Ticker symbol for the order (default: 'TQQQ').
    """
    qty = 25
    side = 'buy'
    type = 'limit'
    time_in_force = 'gtc'
    stop_price = 50

    test_obj = alpaca_api.AlpacaConnect(api_key, secret_key)
    result = test_obj.place_order(ticker, qty, side, type, time_in_force, stop_price)

    print(
        f'Symbol: {result["symbol"]} '
        f'| Side {result["side"]} '
        f'| Qty: {result["qty"]} '
        f'| Order Type: {result["type"]} '
        f'| Status: {result["status"]} '
        f'| Order Id: {result["id"]}'
    )

def test_cancel_order():
    """
    Test to cancel a specific order by its order ID and print the result.
    """
    order_id = '1'

    test_obj = alpaca_api.AlpacaConnect(api_key, secret_key)
    result = test_obj.cancel_one_order(order_id)
    
    print(result)

def test_cancel_all_orders():
    """
    Test to cancel all open orders and print the result.
    """
    test_obj = alpaca_api.AlpacaConnect(api_key, secret_key)
    result = test_obj.cancel_all_orders()

    print(result)

def test_close_position():
    """
    Test to close a position in a specific ticker symbol and print the result.

    Args:
    ticker (str): Ticker symbol for the position to close (default: 'TQQQ').
    """
    ticker = 'TQQQ'
    test_obj = alpaca_api.AlpacaConnect(api_key, secret_key)
    result = test_obj.close_position(ticker)

    print(result)

def test_get_portfolio_history():
    """
    Test to retrieve and print the portfolio history.
    """
    test_obj = alpaca_api.AlpacaConnect(api_key, secret_key)
    result = test_obj.get_portfolio_history()

    print(result)

def test_get_activity():
    """
    Test to retrieve and print account trade activity.
    """
    test_obj = alpaca_api.AlpacaConnect(api_key, secret_key)
    result = test_obj.get_activity()

    print(result)


if __name__ == '__main__':
    print('test_get_account_summary')
    test_get_account_summary()
    print('test_get_all_open_positions')
    test_get_all_open_positions()
    print('test_place_order')
    test_place_order()
    print('test_get_orders')
    test_get_orders()
    # print('test_stop_loss_order')
    # test_stop_loss_order()
    print('test_limit_order')
    test_limit_order()
    print('test_cancel_order')
    test_cancel_order()
    print('test_cancel_all_orders')
    test_cancel_all_orders()
    print('test_close_position')
    test_close_position()
    print('test_history')
    test_history()
    print('test_get_portfolio_history')
    test_get_portfolio_history()
    print('test_get_activity')
    test_get_activity()

