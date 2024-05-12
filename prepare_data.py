import torch
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler


# Date Ranges
# Data is pulled from 1/1/2011 to 12/31/2023, but must factor in holidays and weekends
# Model must be trained on stock measures calculated based on adjusted closing prices
# We will execute Buy/Sell/Hold orders on the next day's open price
# Therefore, the adjusted closing prices we use to calculate the stock measures the
# model is trained on will be offset by 1 day from the openning prices we execute on as follows:

# Open/Close Training Series
# self.close_prices_training_start_date = datetime.datetime(2011, 1, 3)  # Market was not open 1/1 and 1/2
# self.open_prices_training_start_date = datetime.datetime(2011, 1, 4)
# self.close_prices_training_end_date = datetime.datetime(2021, 12, 30)
# self.open_prices_training_end_date = datetime.datetime(2021, 12, 31)  # Friday

# Open/Close Validation Series
# self.close_prices_validation_start_date = datetime.datetime(2022, 1, 3)  # Market was not open 1/1 and 1/2
# self.open_prices_validation_start_date = datetime.datetime(2022, 1, 4)
# self.close_prices_validation_end_date = datetime.datetime(2023, 12, 28)
# self.open_prices_validation_end_date = datetime.datetime(2023, 12, 29)  # Friday


class DataCollector:
    """
    Represents an object that handles raw stock data. Executes stock metric calculations, drops unneeded columns, and
    creates a tensor to be used in training or testing.
    """
    def __init__(self, df: pd.DataFrame, model_dates):
        # Input data
        self.df = df
        self.closing_prices = None
        self.opening_prices = None
        # Window size and time shift in days, used for calculations
        self.windows = [16, 32, 64]
        self.time_shifts = [2, 4, 6, 8, 10]
        # Output tensor
        self.data_tensor = torch.tensor([])
        self.data_shape = None
        # need to split data into training and testing
        self.training_tensor = torch.tensor([])
        self.training_prices = None
        self.testing_tensor = torch.tensor([])
        self.testing_prices = None
        # Unwanted columns
        columns_to_drop_before_normalization = ["ticker"]
        columns_to_drop_after_backfill = ["open", "high", "low", "close", "volume", "adjclose"]
        # Prepare and calculate data
        self._clean_data()
        # print("DataCollector._clean_data():", self.df)
        self._calculate_stock_measures()
        # print("DataCollector._calculate_stock_measures():", self.df)
        self.df.drop(columns_to_drop_before_normalization, axis=1, inplace=True)
        self._normalize_data()
        # print("DataCollector._normalize_data():", self.df)
        self._backfill_data()
        # print("DataCollector._backfill_data():", self.df)
        # Drop unwanted columns
        self.df.drop(columns_to_drop_after_backfill, axis=1, inplace=True)
        # print("DataCollector.drop():", self.df)
        self.df.to_csv("prepared_data.csv", index=True)
        # Split data into training and testing sets
        self._partition_data(model_dates.close_prices_training_end_date, model_dates)

    def _clean_data(self) -> None:
        """
        Set index as the timestamp and drop rows with missing values.
        """
        # Set index as timestamp
        self.df.rename(columns={"date": "timestamp"}, inplace=True)
        self.df = self.df.set_index("timestamp")
        # Finds rows without any values in open, high, low, close, adjclose, volume
        rows_with_missing_values = self.df[self.df.isnull().all(axis=1)]
        # Finds rows missing a timestamp
        index_missing = self.df[self.df.index.isnull()]
        rows_to_drop = pd.concat([rows_with_missing_values, index_missing])
        # Removes rows where all values are missing or just the timestamp is missing
        self.df = self.df.drop(rows_to_drop.index)
        self.closing_prices = self.df["adjclose"]
        self.opening_prices = self.df["open"]

    def _calculate_stock_measures(self):
        """
        Calculates velocity, acceleration, and average true range.
        """
        # Create stock measurements
        for window in self.windows:
            # Add velocity data to self.df
            self._create_velocity_data(window)
            for time_shift in self.time_shifts:
                self._create_velocity_data(window, time_shift)
            # Add acceleration data to self.df
            self._create_acceleration_data(window)
            for time_shift in self.time_shifts:
                self._create_acceleration_data(window, time_shift)
            # Add average true range to self.df
            self._create_avg_true_range_data(window)
            for time_shift in self.time_shifts:
                self._create_avg_true_range_data(window, time_shift)

    def _weighted_moving_avg(self, close_series: pd.Series, window: int) -> pd.Series:
        """
        Calculate weights to create weighted moving average values.
        """
        # Define weights
        weights = np.arange(1, window + 1)
        # Calculate and return the weighted moving average for the passed close series
        return close_series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    def _hull_moving_avg(self, close_series: pd.Series, window: int) -> pd.Series:
        """
        Creates hull moving average series to be added to self.df.
        """
        # Calculate first term
        weighted_half_window = self._weighted_moving_avg(close_series, window // 2)
        # Calculate the second term
        weighted_full_window = self._weighted_moving_avg(close_series, window)
        # Combine the terms and final calculation
        hma_series = 2 * weighted_half_window - weighted_full_window
        hma_series = pd.Series(self._weighted_moving_avg(hma_series, int(np.sqrt(window))), index=close_series.index)
        return hma_series.dropna()

    def _create_velocity_data(self, window: int, time_shift: int = 0) -> None:
        """
        Calculate velocity data and add the series to self.df.
        """
        # Branch for window without time shift
        if time_shift == 0:
            close_series = self.df['adjclose']
            hma_series = self._hull_moving_avg(close_series, window)
            log_hma_series = pd.Series(np.log(hma_series), index=hma_series.index)
            new_series = log_hma_series.diff()
        # Branch if the data should be time_shifted
        else:
            new_series = pd.Series(self.df[f"velocity_{window}w_0ts"].shift(time_shift), index=self.df.index)
        # Drop na values from series and add series to self.df
        new_series.dropna(inplace=True)
        self.df[f"velocity_{window}w_{time_shift}ts"] = new_series

    def _create_acceleration_data(self, window: int, time_shift: int = 0):
        """
        Calculate acceleration data based on the velocity data and add it to self.df.
        """
        # Branch for window without time shift
        if time_shift == 0:
            new_series = pd.Series(self.df[f"velocity_{window}w_{time_shift}ts"].diff(), index=self.df.index)
        # Branch if time shift is present
        else:
            new_series = pd.Series(self.df[f"acceleration_{window}w_0ts"].shift(time_shift),
                                   index=self.df.index)
        # Drop na values and add series to self.df
        new_series.dropna(inplace=True)
        self.df[f"acceleration_{window}w_{time_shift}ts"] = new_series

    def _create_avg_true_range_data(self, window: int, time_shift: int = 0):
        """
        Calculate the average true range and add the data to self.df
        """
        # Branch if window without time shift
        if time_shift == 0:
            # Collect needed data series
            data_index = self.df.index
            high_series = self.df['high']
            low_series = self.df['low']
            close_prev_series = self.df['adjclose'].shift(1)
            # Calculate the true range
            true_range = (
                pd.DataFrame({
                    'h_l': high_series - low_series,
                    'h_c_prev': abs(high_series - close_prev_series),
                    'l_c_prev': abs(low_series - close_prev_series)
                }, index=data_index)
                .max(axis=1)
            )
            # Convert true range to the average true range
            true_range_series = self._hull_moving_avg(true_range, window)
        # Branch if time shift is present
        else:
            true_range_series = self.df[f"atr_{window}w_0ts"].shift(time_shift)
        # Add series to self.df
        self.df[f"atr_{window}w_{time_shift}ts"] = true_range_series

    def _normalize_data(self):
        """
        Normalize all values except for the timestamp column. Min-max scaling to normalize the data between 0 and 1.
        """
        # Extract the timestamp column
        timestamp_column = self.df.index
        # Convert the self.df to a NumPy array
        data_array = self.df.values
        # Normalize all data columns except for the timestamp column
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(data_array)
        # Recreate the DataFrame with the normalized data
        self.df = pd.DataFrame(normalized_data, index=timestamp_column, columns=self.df.columns)

    def _backfill_data(self):
        """
        Backfills cells that do not have a value.
        """
        for column in self.df.columns:
            self.df[column] = self.df[column].interpolate(method='linear')
        # Catches any remaining NaN cells
        self.df = self.df.bfill().ffill().fillna(0)

    def _partition_data(self, split_index: datetime.date, model_dates) -> None:
        """
        Partitions the data into training and testing sets.
        """
        # Convert the normalized dataframe to a tensor
        self.data_tensor = torch.tensor(self.df.values, dtype=torch.float32)
        tensor_df = pd.DataFrame(self.data_tensor)
        tensor_df.to_csv("tensor_df.csv", index=True)

        # Get the integer location of start and end dates
        close_prices_training_start_index = self.df.index.get_loc(model_dates.close_prices_training_start_date)
        close_prices_training_end_index = self.df.index.get_loc(model_dates.close_prices_training_end_date)
        open_prices_training_start_index = self.df.index.get_loc(model_dates.open_prices_training_start_date)
        open_prices_training_end_index = self.df.index.get_loc(model_dates.open_prices_training_end_date)

        close_prices_validation_start_index = self.df.index.get_loc(model_dates.close_prices_validation_start_date)
        close_prices_validation_end_index = self.df.index.get_loc(model_dates.close_prices_validation_end_date)
        open_prices_validation_start_index = self.df.index.get_loc(model_dates.open_prices_validation_start_date)
        open_prices_validation_end_index = self.df.index.get_loc(model_dates.open_prices_validation_end_date)

        print("close_prices_training_start_index", close_prices_training_start_index)
        print("close_prices_training_end_index", close_prices_training_end_index)
        print("open_prices_training_start_index", open_prices_training_start_index)
        print("open_prices_training_end_index", open_prices_training_end_index)

        print("close_prices_validation_start_index", close_prices_validation_start_index)
        print("close_prices_validation_end_index", close_prices_validation_end_index)
        print("open_prices_validation_start_index", open_prices_validation_start_index)
        print("open_prices_validation_end_index", open_prices_validation_end_index)

        # Use these indexes to slice your data
        self.training_tensor = torch.tensor(self.df.iloc[close_prices_training_start_index:close_prices_training_end_index + 1].values, dtype=torch.float32)
        # self.training_tensor = torch.tensor(self.df.loc[:split_index].values, dtype=torch.float32)
        training_tensor_df = pd.DataFrame(self.training_tensor)
        training_tensor_df.to_csv("training_tensor.csv", index=True)

        # self.training_prices = self.closing_prices.loc[:split_index]

        self.training_prices = self.closing_prices.iloc[close_prices_training_start_index:close_prices_training_end_index + 1]
        self.training_prices.to_csv("training_prices.self.closing_prices.iloc[close_prices_training_start_index:close_prices_training_end_index + 1].csv", index=True)

        # self.testing_tensor = torch.tensor(self.df.loc[split_index:].values, dtype=torch.float32)
        close_prices_training_start_index = self.df.index.get_loc(model_dates.close_prices_training_start_date)
        close_prices_training_end_index = self.df.index.get_loc(model_dates.close_prices_training_end_date)
        self.testing_tensor = torch.tensor(self.df.iloc[close_prices_training_start_index:close_prices_training_end_index + 1].values, dtype=torch.float32)

        # print("self.testing_tensor:", self.testing_tensor)
        testing_tensor_df = pd.DataFrame(self.testing_tensor)
        testing_tensor_df.to_csv("testing_tensor_df.csv", index=True)

        self.testing_prices = self.closing_prices.loc[split_index:]
        self.testing_prices.to_csv("testing_prices.self.closing_prices.loc[split_index:].csv", index=True)
        # print("self.testing_prices:", self.testing_prices)

        self.training_prices = self.opening_prices
        # print("opening self.training_prices:", self.training_prices)
        self.testing_prices = self.opening_prices.loc[split_index:]
        # print("opening self.testing_prices:", self.testing_prices)
