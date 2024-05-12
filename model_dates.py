import datetime


class ModelDates:
    def __init__(self):

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
