import datetime


class ModelDates:
    def __init__(self, training_start_date: datetime.datetime):
        # self.training_start_date = training_start_date.date()
        # self.testing_end_date = datetime.datetime.now().date()
        # self.training_end_date = (self.testing_end_date - datetime.timedelta(days=730))
        # self.testing_start_date = (self.training_end_date + datetime.timedelta(days=1))
        self.training_start_date = datetime.datetime(2011, 1, 1)
        self.training_end_date = datetime.datetime(2021, 12, 31)
        self.testing_start_date = datetime.datetime(2022, 1, 1)
        self.testing_end_date = datetime.datetime(2023, 12, 31)
