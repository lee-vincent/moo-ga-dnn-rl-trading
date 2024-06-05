import datetime


class ModelDates:
    def __init__(self):

        # The indicators calculated in prepare_data use up to 64 days of previous adjclose price in calculations.
        # Therefore we need to pull data from date ranges before the actual training period to provide data we
        # can use to warm up our indicator calculations.

        # Inference Date
        self.inference_date = None

        # Indicator Warm-Up Dates
        self.indicator_warmup_start_date = None  # this needs to be set by fetch_data because every ticker has a different debut date on the exchange
        self.indicator_warmup_end_date = datetime.datetime(2010, 12, 31)  # hardcoded as the trading day immediately preceding close_prices_training_start_date

        # Open/Close Training Series
        self.close_prices_training_start_date = datetime.datetime(2011, 1, 3)  # Market was not open 1/1 and 1/2
        self.open_prices_training_start_date = datetime.datetime(2011, 1, 4)
        self.close_prices_training_end_date = datetime.datetime(2021, 12, 30)
        self.open_prices_training_end_date = datetime.datetime(2021, 12, 31)  # Friday

        # Open/Close Validation Series
        self.close_prices_validation_start_date = datetime.datetime(2022, 1, 3)  # Market was not open 1/1 and 1/2
        self.open_prices_validation_start_date = datetime.datetime(2022, 1, 4)
        self.close_prices_validation_end_date = datetime.datetime(2023, 12, 28)
        self.open_prices_validation_end_date = datetime.datetime(2023, 12, 29)  # Friday

    def set_indicator_warmup_start_date(self, start_date):
        self.indicator_warmup_start_date = start_date

    def set_inference_date(self, inference_date):
        self.inference_date = inference_date
