import datetime
import argparse


def get_previous_market_close_date(today):
    """ Returns the previous market close date based on whether today is a market open day. """
    market_open_days = {0, 1, 2, 3, 4}  # Monday to Friday
    while today.weekday() not in market_open_days:
        today -= datetime.timedelta(days=1)
    return today


def calculate_date_ranges(input_date):
    # Determine the last valid market close date from the input date
    last_valid_market_date = get_previous_market_close_date(input_date)

    # Set the fixed start date for the data
    start_date = datetime.datetime(2011, 1, 1)

    # Determine the end of the training date range as exactly one year before the last valid market date
    end_training_date = last_valid_market_date - datetime.timedelta(days=365)

    # The testing range starts the day after the training ends
    start_testing_date = end_training_date + datetime.timedelta(days=1)

    # Adjust the end of the testing date to be the last market day before the last valid market date
    # If the last valid market date is a Monday, adjust to end on the previous Friday
    end_testing_date = last_valid_market_date - datetime.timedelta(days=1)
    if last_valid_market_date.weekday() == 0:  # Monday
        end_testing_date -= datetime.timedelta(days=2)  # Set to previous Friday

    return {
        "Stock Data Range": f"{start_date.date()} -> {last_valid_market_date.date()}",
        "Training Tensor Range": f"{start_date.date()} -> {end_training_date.date()}",
        "Testing Tensor Range": f"{start_testing_date.date()} -> {end_testing_date.date()}",
        "Day to Run Inference On": f"{last_valid_market_date.date()}"
    }


def main():
    parser = argparse.ArgumentParser(description='Calculate date ranges for stock data analysis.')
    parser.add_argument('date', type=str, help='The current date in YYYY-MM-DD format')
    args = parser.parse_args()

    # Convert the provided date string to a datetime object
    input_date = datetime.datetime.strptime(args.date, '%Y-%m-%d')

    # Calculate the date ranges based on the provided date
    date_ranges = calculate_date_ranges(input_date)
    for key, value in date_ranges.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
