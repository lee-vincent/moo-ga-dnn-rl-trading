from pathlib import Path
import argparse
import torch
import pandas as pd
from torch.nn import DataParallel
from prepare_data import DataCollector
from trading_problem import TradingProblem, PerformanceLogger
from policy_network import PolicyNetwork
from trading_environment import TradingEnvironment
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
        default="./inference_candidates/TQQQ/ngen_600/npop_100/model_0_2024-05-14_21-12-48.pt",
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
    This should be run daily after the market closes at 4PM Easter Time for a valid inference.
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

    exit()
