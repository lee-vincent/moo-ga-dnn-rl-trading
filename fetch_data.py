from yahoo_fin import stock_info as si
import datetime


def fetch_data(ticker: str, start_date: datetime.date, end_date: datetime.date, save: bool = False):
    """
    Downloads historical price data from Yahoo! Finance of a stock into a pandas data frame.
    Offers the functionality to pull daily, weekly, or monthly data.
    The 'ticker' column is dropped and the data frame can be saved as a CSV.
    """
    df = si.get_data(ticker=ticker, start_date=start_date, end_date=end_date, index_as_date=False, interval="1d")

    if save:
        ct = datetime.datetime.now()
        ft = ct.strftime("%Y-%m-%d_%H-%M-%S")
        df.to_csv(f"raw_{ticker}_data_{ft}.csv", index=False)

    return df
