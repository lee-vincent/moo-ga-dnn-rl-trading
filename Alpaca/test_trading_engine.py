import trading_engine
import sys
import config

def main(alpaca_key, alpaca_secret_key, recs, amount_to_invest=0.1, paper=True, run_locally=False):
    """
    Main function to initialize and run the trading engine.
    
    Args:
    - alpaca_key (str): API key for Alpaca.
    - alpaca_secret_key (str): Secret key for Alpaca.
    - recs (dict): Dictionary of stock recommendations and share prices.
    - amount_to_invest (float): Percentage of total cash to invest in each asset (default is 0.1).
    - paper (bool): Whether to use paper trading or real trading (default is True).
    - run_locally (bool): Whether to run the trading engine locally or not (default is False).
    """
    obj = trading_engine.tradingEngine(alpaca_key, alpaca_secret_key, recs, amount_to_invest, run_locally)
    r = obj.main()
    print(r)

if __name__ == '__main__':
    args = sys.argv[1:]

    arg3 = {
        'TQQQ': {'recommendation': 'sell', 'share_price': 42.16},
        'SSO': {'recommendation': 'sell', 'share_price': 59.68},
        'IWM': {'recommendation': 'buy', 'share_price': 194.3},
        'EEM': {'recommendation': 'buy', 'share_price': 40.48}
    }

    main(config.ALPACA_KEY, config.ALPACA_SECRET_KEY, arg3, 0.1, True, False)
