from yahoo_fin import stock_info


def get_data(ticker: str, save: bool = False):
    """
    Downloads historical price data from Yahoo! Finance of a stock into a pandas data frame.
    Offers the functionality to pull daily, weekly, or monthly data.
    The 'ticker' column is dropped and the data frame can be saved as a CSV.
    """

    ticker = ticker.lower()
    raw_df = stock_info.get_data(ticker=ticker, start_date="2011-01-01", end_date="2023-12-31", index_as_date=False, interval="1d")
    raw_df.rename(columns={"date": "timestamp"}, inplace=True)
    raw_df.drop(columns=["ticker"], inplace=True)
    # we should really validate that this function actually returned data for each day. if a column is missing data,
    # then we should interpolate a value for it using something like bfill, ffill and interpolation
    if save:
        raw_df.to_csv(f"raw_{ticker}_data.csv", index=False)
    else:
        return raw_df
