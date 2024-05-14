from yahoo_fin import stock_info as si
import datetime


def fetch_data(ticker: str, model_dates, save: bool = False):
    """
    Downloads historical price data from Yahoo! Finance of a stock into a pandas data frame.
    Offers the functionality to pull daily, weekly, or monthly data.
    """

    try:
        # get_data is not end date inclusive, so have to add 1 day to get correct range
        df = si.get_data(ticker=ticker, end_date=(model_dates.open_prices_validation_end_date+datetime.timedelta(days=1)), index_as_date=False, interval="1d")
    except AssertionError:
        # yahoo_fin could not find data. Return 1 for main.py to exit program
        print("Error retrieving stock data. Check ticker symbol.")
        return None

    print(f"First available date for {ticker}: {df.date[0].to_pydatetime()}")
    model_dates.set_indicator_warmup_start_date(df.date[0].to_pydatetime())

    if save:
        ct = datetime.datetime.now()
        ft = ct.strftime("%Y-%m-%d_%H-%M-%S")
        df.to_csv(f"raw_{ticker}_data_{ft}.csv", index=False)

    return df
