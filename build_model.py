import argparse
import datetime
from pathlib import Path
from model_dates import ModelDates
from prepare_data import DataCollector
from set_path import set_path
import pandas as pd
from plotter import Plotter
from fetch_data import fetch_data
import torch
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.problem import StarmapParallelization
from pymoo.optimize import minimize
from trading_environment import TradingEnvironment
import multiprocessing as mp
from torch.nn import DataParallel
from policy_network import PolicyNetwork
from trading_problem import TradingProblem, PerformanceLogger


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
        '--profit_threshold',
        type=float,
        default=100.0,
        help='Profit threshold for considering a model worth saving'
    )
    parser.add_argument(
        '--drawdown_threshold',
        type=float,
        default=40.0,
        help='Drawdown threshold for considering a model worth saving'
    )
    parser.add_argument(
        '--ticker',
        type=str,
        default="TQQQ",
        help='Ticker symbol for the stock data'
    )
    parser.add_argument(
        '--training_start_date',
        type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'),
        default=datetime.datetime(2011, 1, 1),
        help='The date in the past the model will be trained from in YYYY-MM-DD format')
    parser.add_argument(
        '--save_data',
        action='store_true',  # This sets the flag to True if it is present.
        default=False,
        help='Save the raw dataset to csv: open, high, low, close, adjclose, volume'
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


def train_and_validate(queue, n_pop, n_gen, prepared_data):
    SCRIPT_PATH = Path(__file__).parent
    # Create the neural network
    network = PolicyNetwork([prepared_data.data_tensor.shape[1], 64, 32, 16, 8, 4, 3])
    if torch.cuda.device_count() > 1:
        # Use DataParallel to use multiple GPUs
        network = DataParallel(network)
    network.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    trading_env = TradingEnvironment(prepared_data.training_tensor, network, prepared_data.training_prices)

    # initialize the thread pool and create the runner for ElementwiseProblem parallelization
    n_threads = 4
    pool = mp.pool.ThreadPool(n_threads)
    runner = StarmapParallelization(pool.starmap)

    # Create the trading problem
    problem = TradingProblem(prepared_data.training_tensor, network, trading_env, elementwise_runner=runner)
    # Create the algorithm
    algorithm = NSGA2(
        pop_size=n_pop,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.2, eta=20),
        eliminate_duplicates=True
    )
    performance_logger = PerformanceLogger(queue)

    # Run the optimization
    res = minimize(
        problem,
        algorithm,  # this calls TradingProblem._evaluate which calls TradingEnvironment.simulate_trading
        ('n_gen', n_gen),
        callback=performance_logger,
        verbose=True,
        save_history=True
    )

    date_time = pd.to_datetime("today").strftime("%Y-%m-%d_%H-%M-%S")
    history: pd.DataFrame = pd.DataFrame(performance_logger.history)
    history.to_csv(set_path(SCRIPT_PATH, f"Output/performance_log/ngen_{n_gen}", f"{date_time}.csv"))

    generations = history["generation"].values
    objectives = history["objectives"].values
    # VL: flake8 is complaining that decisions is never accessed, so why is it defined here?
    decisions = history["decision_variables"].values
    print("Decisions:", decisions)
    best = history["best"].values

    historia = []

    for i in range(len(generations)):
        avg_profit, avg_drawdown, avg_trades = 0, 0, 0
        objs = objectives[i]
        for row in objs:
            avg_profit += row[0]
            avg_drawdown += row[1]
            avg_trades += row[2]
        avg_profit /= len(objs)
        avg_drawdown /= len(objs)
        avg_trades /= len(objs)
        row = [generations[i], avg_profit, avg_drawdown, avg_trades, best[i]]
        historia.append(row)

    history_df: pd.DataFrame = pd.DataFrame(
        columns=["generation", "avg_profit", "avg_drawdown", "num_trades", "best"],
        data=historia
    )
    history_df.to_csv(set_path(SCRIPT_PATH, f"Output/performance_log/ngen_{n_gen}", f"{date_time}_avg.csv"))

    trading_env.set_features(prepared_data.testing_tensor)
    trading_env.set_closing_prices(prepared_data.testing_prices)
    population = None if res.pop is None else res.pop.get("X")

    validation_results = []
    max_ratio = 0.0
    best_network = None
    if population is not None:
        for i, x in enumerate(population):
            # VL: does map_params_to_model really need to be its own function? hard to tell what's going on
            map_params_to_model(network, x)
            # VL: why did previous team comment this out?
            # torch.save(network.state_dict(), f"Output/policy_networks/{date_time}_ngen_{n_gen}_top_{i}.pt")
            trading_env.reset()
            profit, drawdown, num_trades = trading_env.simulate_trading()
            ratio = profit / drawdown if drawdown != 0 else profit / 0.0001

            if ratio > max_ratio and drawdown < 55.0:
                best = ratio
                best_network = network.state_dict()

            validation_results.append([profit, drawdown, num_trades, ratio, str(x)])
        torch.save(best_network, set_path(SCRIPT_PATH, f"Output/policy_networks/ngen_{n_gen}", f"{date_time}_best.pt"))

        queue.put(validation_results)

        validation_results_df = pd.DataFrame(
            columns=["profit", "drawdown", "num_trades", "ratio", "chromosome"],
            data=validation_results
        )

        # sort by ratio
        validation_results_df = validation_results_df.sort_values(by="ratio", ascending=False)

        validation_results_df.to_csv(set_path(SCRIPT_PATH, f"Output/validation_results/ngen_{n_gen}", f"{date_time}.csv"))

    pool.close()


if __name__ == "__main__":
    """
    Used to build and train an ML model optimized with NSGA-II to maximize profit and
    minimize drawdown in the context of trading stocks.
    """

    args = parse_args()

    model_dates = ModelDates(args.training_start_date)

    print(f"Population size: {args.pop_size}")
    print(f"Number of generations: {args.n_gen}")
    print(f"Profit threshold: {args.profit_threshold}")
    print(f"Drawdown threshold: {args.drawdown_threshold}")
    print(f"Ticker symbol: {args.ticker}")
    print(f"Training Start Date: {model_dates.training_start_date}")
    print(f"Training End Date: {model_dates.training_end_date}")
    print(f"Testing Start Date: {model_dates.testing_start_date}")
    print(f"Testing End Date: {model_dates.testing_end_date}")
    print(f"Save Data: {args.save_data}")

    # Fetch historical data
    data = fetch_data(args.ticker, model_dates.training_start_date, model_dates.testing_end_date, args.save_data)
    prepared_data = DataCollector(data, model_dates.training_end_date)

    queue = mp.Queue()
    plotter = Plotter(queue, args.n_gen)
    train_and_validate_process = mp.Process(target=train_and_validate, args=(queue, args.pop_size, args.n_gen, prepared_data))
    train_and_validate_process.start()
    plotter.update_while_training()
    train_and_validate_process.join()
    train_and_validate_process.close()
    queue.close()

    exit()
