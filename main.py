from pathlib import Path
import argparse
import torch
import pandas as pd
from torch.nn import DataParallel
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
import multiprocessing as mp
import multiprocessing.pool
from pymoo.core.problem import StarmapParallelization
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


IS_TRAINING = True


def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Run trading optimization with optional settings."
    )

    # Add arguments
    parser.add_argument(
        '--pop_size',
        type=int,
        default=100,
        help='Population size for the genetic algorithm'
    )
    parser.add_argument(
        '--n_gen',
        type=int,
        default=100,
        help='Number of generations for the genetic algorithm'
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


def train_and_validate(queue, n_pop, n_gen, data, model_dates, force_cpu, ticker):

    SCRIPT_PATH = Path(__file__).parent

    # Manipulate data and calculate stock measures
    prepared_data = DataCollector(data, model_dates)

    # Create the policy network
    print("prepared_data.data_tensor.shape[1]:", prepared_data.data_tensor.shape[1])
    network = PolicyNetwork([prepared_data.data_tensor.shape[1], 64, 32, 16, 8, 4, 3])

    timestamped_print(f"CUDA available? {torch.cuda.is_available()}")

    # Check if multiple GPUs are available
    if force_cpu:
        timestamped_print("Force CPU.")
        network.to(torch.device("cpu"))
    elif torch.cuda.device_count() > 1:
        timestamped_print(f"{torch.cuda.device_count()} GPUs available.")
        network = DataParallel(network)  # Use DataParallel to use multiple GPUs
        network.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    elif torch.cuda.device_count() == 1:
        timestamped_print("Using a single GPU.")
        network.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        timestamped_print("No GPUs available. Using CPU.")
        network.to(torch.device("cpu"))

    # Create the trading environment
    timestamped_print("creating trading environment")
    trading_env = TradingEnvironment(
        prepared_data.training_tensor,
        network,
        prepared_data.training_open_prices,
        force_cpu)

    # initialize the thread pool and create the runner for ElementwiseProblem parallelization
    n_threads = os.cpu_count()
    pool = mp.pool.ThreadPool(n_threads)
    runner = StarmapParallelization(pool.starmap)

    timestamped_print("Create the trading problem")
    # Create the trading problem
    problem = TradingProblem(network, trading_env, elementwise_runner=runner)

    timestamped_print("Create the algorithm")
    # Create the algorithm
    algorithm = NSGA2(
        pop_size=n_pop,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.2, eta=20),
        eliminate_duplicates=True
    )

    timestamped_print("performance_logger = PerformanceLogger(queue)")
    performance_logger = PerformanceLogger(queue)

    timestamped_print("Run the optimization")
    # Run the optimization
    res = minimize(
        problem,
        algorithm,  # this calls TradingProblem._evaluate which calls TradingEnvironment.simulate_trading
        ('n_gen', n_gen),
        callback=performance_logger,
        verbose=True,
        save_history=True
    )

    timestamped_print("Optimization Completed")
    date_time = pd.to_datetime("today").strftime("%Y-%m-%d_%H-%M-%S")

    X = res.X

    # below is where test/validation happens - we should already have the pareto set from above.
    trading_env.set_features(prepared_data.validation_tensor)
    trading_env.set_opening_prices(prepared_data.validation_open_prices)
    validation_results = []
    for i, _x in enumerate(X):
        map_params_to_model(network, _x)
        trading_env.reset()
        profit, drawdown = trading_env.simulate_trading()
        timestamped_print(f"Profit: {profit}, Drawdown: {drawdown}")
        if profit > 0.0:
            torch.save(network.state_dict(), set_path(SCRIPT_PATH, f"inference_candidates/{ticker}/ngen_{n_gen}/npop_{n_pop}/", f"model_{i}_profit_{profit:.2f}_drawdown_{drawdown:.2f}_{date_time}.pt"))
        validation_results.append([profit, drawdown])

    queue.put(validation_results)

    pool.close()


if __name__ == '__main__':
    """
    Basic multi-objective optimization using NSGA-II.
    This main script is used to run the optimization.
    Can be run in a python notebook or as a standalone script.
    """

    args = parse_args()

    model_dates = ModelDates()

    # Get stock data from yahoo_fin
    stock_data = fetch_data(args.ticker, model_dates, IS_TRAINING, args.save_data)
    if stock_data is None:
        # yahoo_fin could not find data. Exit program
        sys.exit(1)

    # NSGA-II Algorithm Hyper-Parameters
    timestamped_print(f"Population size: {args.pop_size}")
    timestamped_print(f"Number of generations: {args.n_gen}")
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

    queue = mp.Queue()
    plotter = Plotter(queue, args.n_gen)

    timestamped_print("Creating process: train_and_validate_process")
    train_and_validate_process = mp.Process(target=train_and_validate, args=(queue,
                                                                             args.pop_size,
                                                                             args.n_gen,
                                                                             stock_data,
                                                                             model_dates,
                                                                             args.force_cpu,
                                                                             args.ticker))

    timestamped_print("train_and_validate_process.start()")
    train_and_validate_process.start()

    timestamped_print("plotter.update_while_training()")
    plotter.update_while_training()

    timestamped_print("train_and_validate_process.join()")
    train_and_validate_process.join()

    timestamped_print("train_and_validate_process.close()")
    train_and_validate_process.close()

    timestamped_print("queue.close()")
    queue.close()

    timestamped_print("Training and validation process finished.")
