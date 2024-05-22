from yahoo_fin import stock_info as si
import datetime


def fetch_data(ticker: str, model_dates, _IS_TRAINING: bool = True, save: bool = False):
    """
    Downloads historical price data from Yahoo! Finance of a stock into a pandas data frame.
    Offers the functionality to pull daily, weekly, or monthly data.
    """

    if _IS_TRAINING:
        _end_date_requested = model_dates.open_prices_validation_end_date+datetime.timedelta(days=1)
    else:
        _end_date_requested = datetime.datetime.now().date()

    # print("_end_date_requested:", _end_date_requested)

    try:
        # get_data is not end date inclusive, so have to add 1 day to get correct range
        df = si.get_data(ticker=ticker, end_date=_end_date_requested, index_as_date=False, interval="1d")
    except AssertionError:
        # yahoo_fin could not find data. Return 1 for main.py to exit program
        print("Error retrieving stock data. Check ticker symbol.")
        return None

    model_dates.set_indicator_warmup_start_date(df.date[0].to_pydatetime())

    if not _IS_TRAINING:
        _end_date_received = df['date'].iloc[-1].date()
        # print("_end_date_received:", _end_date_received)
        if (_end_date_requested != _end_date_received):
            print(f"WARNING: Close price for {ticker} not available for {_end_date_requested}. Is the market still open or is it a holiday?")
            print(f"WARNING: Inference will be performed for {_end_date_received} close price.")
    if save:
        ct = datetime.datetime.now()
        ft = ct.strftime("%Y-%m-%d_%H-%M-%S")
        df.to_csv(f"raw_{ticker}_data_{ft}.csv", index=False)

    return df
