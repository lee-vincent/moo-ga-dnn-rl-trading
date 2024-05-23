from pathlib import Path
import argparse
import torch
import pandas as pd
from prepare_data import DataCollector
from policy_network import PolicyNetwork
from plotter import Plotter
import os
from timestamped_print import timestamped_print
from model_dates import ModelDates
from fetch_data import fetch_data
from set_path import set_path
import sys
import matplotlib.pyplot as plt


IS_TRAINING = False


def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Run trading inference with optional settings."
    )

    # Add arguments
    parser.add_argument(
        '--model',
        type=str,
        default="./inference_candidates/TQQQ/ngen_300/npop_200/model_23_profit_282.19_drawdown_31.14_2024-05-22_20-01-31.pt",
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
        '--force_cpu',
        action='store_true',  # This sets the flag to True if it is present.
        default=False,
        help='Force training to run on CPU even if a GPU is available'
    )

    # Parse the arguments
    args = parser.parse_args()

    return args


def map_params_to_model(model, params):
    """
    Decodes (i.e. maps) the genes of an individual (x) into the policy network.
    """
    idx = 0  # Starting index in the parameter vector
    new_state_dict = {}  # New state dictionary to load into the model
    for name, param in model.named_parameters():  # Iterate over each layer's weights and biases in the model
        num_param = param.numel()  # Compute the number of elements in this layer
        param_values = params[idx:idx + num_param]  # Extract the corresponding part of `params`
        param_values = param_values.reshape(param.size())  # Reshape the extracted values into the correct shape for this layer
        param_values = torch.Tensor(param_values)  # Convert to the appropriate tensor
        new_state_dict[name] = param_values  # Add to the new state dictionary
        idx += num_param  # Update the index
    model.load_state_dict(new_state_dict)  # Load the new state dictionary into the model


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
    timestamped_print(f"Force CPU: {args.force_cpu}")
    # Inference Model
    timestamped_print(f"Inference Model: {args.model}")

    prepared_data = DataCollector(stock_data, model_dates, IS_TRAINING)

    inference_model = PolicyNetwork([prepared_data.inference_tensor.shape[1], 64, 32, 16, 8, 4, 3])
    inference_model.load_state_dict(torch.load(args.model))
    inference_model.eval()

    trading_decisions = []
    for i in range(len(prepared_data.inference_tensor)):  # this is all the rows in  training_tqqq_prepared.csv
        feature_vector = prepared_data.inference_tensor[i:i+1]  # Get the feature vector for the current day
        feature_vector = feature_vector.to(torch.device("cpu"))
        decision = inference_model(feature_vector).argmax().item()  # 0=buy, 1=hold, 2=sell
        trading_decisions.append(decision)

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
