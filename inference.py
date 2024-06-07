import argparse
import torch
from prepare_data import DataCollector
from policy_network import PolicyNetwork
from timestamped_print import timestamped_print
from model_dates import ModelDates
from fetch_data import fetch_data
import sys


IS_TRAINING = False


def validate_activation_function(value):
    valid_functions = ['ReLu', 'Tanh']
    if value not in valid_functions:
        raise argparse.ArgumentTypeError("Activation function must be either 'ReLu' or 'Tanh'")
    return value


def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Run trading inference with optional settings."
    )

    # Add arguments
    parser.add_argument(
        '--model',
        type=str,
        default="./inference_candidates/TQQQ/Tanh/ngen_25/npop_100/model_0_profit_329.39_drawdown_31.14_2024-05-24_14-26-05.pt",
        help='Path to model to use for inference'
    )
    parser.add_argument(
        '--ticker',
        type=str,
        default="TQQQ",
        help='Ticker symbol for the stock data'
    )
    parser.add_argument(
        '--save_data',
        action='store_true',  # This sets the flag to True if it is present.
        default=False,
        help='Save the raw dataset to csv: open, high, low, close, adjclose, volume'
    )
    parser.add_argument(
        '--fnc',
        type=validate_activation_function,
        default="Tanh",
        help="Neural Network Activation Function. Options: 'ReLu' or 'Tanh'. Default: 'ReLu'"
    )
    # Parse the arguments
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    """
    This module performs inference on today's close price with a trained PyTorch model.
    A buy, sell, or hold signal is produced and sent via REST api call.
    This should be run daily after the market closes at 4PM Eastern Time for a valid inference.
    Available PyTorch models are available in the ./inference_candidates directory
    """

    args = parse_args()

    model_dates = ModelDates()

    # Get stock data from yahoo_fin
    stock_data = fetch_data(args.ticker, model_dates, IS_TRAINING, args.save_data)
    if stock_data is None:
        # yahoo_fin could not find data. Exit program
        sys.exit(1)

    # Stock Ticker Symbol
    timestamped_print(f"Ticker symbol: {args.ticker}")
    # Close prices to perform inference on
    timestamped_print(f"Inference Date: {model_dates.inference_date}")
    # Indicator Warm-Up Series
    timestamped_print(f"Indicator Warm-Up Start Date: {model_dates.indicator_warmup_start_date}")
    timestamped_print(f"Indicator Warm-Up End Date: {model_dates.indicator_warmup_end_date}")
    # Open/Close Training Series
    timestamped_print(f"Close Prices Training Start Date: {model_dates.close_prices_training_start_date}")
    timestamped_print(f"Open Prices Training Start Date: {model_dates.open_prices_training_start_date}")
    timestamped_print(f"Close Prices Training End Date: {model_dates.close_prices_training_end_date}")
    timestamped_print(f"Open Prices Training End Date: {model_dates.open_prices_training_end_date}")
    # Open/Close Validation Series
    timestamped_print(f"Close Prices Validation Start Date: {model_dates.close_prices_validation_start_date}")
    timestamped_print(f"Open Prices Validation Start Date: {model_dates.open_prices_validation_start_date}")
    timestamped_print(f"Close Prices Validation End Date: {model_dates.close_prices_validation_end_date}")
    timestamped_print(f"Open Prices Validation End Date: {model_dates.open_prices_validation_end_date}")
    # Flags
    timestamped_print(f"Save Data: {args.save_data}")
    # Inference Model
    timestamped_print(f"Inference Model: {args.model}")
    # PyTorch Activation Function
    timestamped_print(f"Activation Function: {args.fnc}")

    prepared_data = DataCollector(stock_data, model_dates, IS_TRAINING)

    inference_model = PolicyNetwork(args.fnc, [prepared_data.inference_tensor.shape[1], 64, 32, 16, 8, 4, 3])
    inference_model.load_state_dict(torch.load(args.model))
    inference_model.eval()

    trading_decisions = []
    for i in range(len(prepared_data.inference_tensor)):  # this is all the rows in  training_tqqq_prepared.csv
        feature_vector = prepared_data.inference_tensor[i:i+1]  # Get the feature vector for the current day
        feature_vector = feature_vector.to(torch.device("cpu"))
        decision = inference_model(feature_vector).argmax().item()  # 0=buy, 1=hold, 2=sell
        trading_decisions.append(decision)

    # import pandas as pd
    # pd.DataFrame(trading_decisions).to_csv("trading_decisions_df.csv", index=True)

    # trade decision for today (model_dates.inference_date)
    trade_decision = trading_decisions[-1]

    if trade_decision == 0:
        print(f"Trade decision for {model_dates.inference_date}: {trade_decision} = buy")
    elif trade_decision == 1:
        print(f"Trade decision for {model_dates.inference_date}: {trade_decision} = hold")
    elif trade_decision == 2:
        print(f"Trade decision for {model_dates.inference_date}: {trade_decision} = sell")
    else:
        print("error: trade_decision should be 0, 1, or 2")
        sys.exit(1)
    exit()
