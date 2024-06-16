import numpy as np
from torch import Tensor
from torch.nn import DataParallel
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.callback import Callback
from trading_environment import TradingEnvironment
from policy_network import PolicyNetwork
# import pandas as pd


class TradingProblem(ElementwiseProblem):
    """
    The trading problem class for the multi-objective optimization.
    Takes a dataset, a policy network, and a trading environment.
    Calculates the number of variables based on the policy network's parameters.
    Calls the superclass constructor with the number of variables, objectives, and lower/upper bounds for x (gene) values.

    *** Still need to add 1 to n_vars for stop-loss gene ***
    """

    def __init__(self, network: DataParallel | PolicyNetwork, environment: TradingEnvironment, *args, **kwargs):
        self.network = network
        self.environment = environment

        # Adjust access to dims based on whether network is wrapped by DataParallel
        if isinstance(network, DataParallel):
            network_dims = network.module.dims
        else:
            network_dims = network.dims

        # n_vars = the total number of weights and biases in the neural network
        # weights and biases are adjusted/mutated in the NSGA-II algorithm?
        # self.n_vars = sum([(network_dims[i] + 1) * network_dims[i + 1] for i in range(len(network_dims) - 1)])
        self.n_vars = 0
        for i in range(len(network_dims) - 1):
            a = (network_dims[i] + 1)
            b = (network_dims[i + 1])
            print(f"a = {a}, b = {b}")
            self.n_vars += a * b
            print(f"i = {i}, self.n_vars = {self.n_vars}")
        # xl=-1.0, xu=1.0 is telling pymoo the lower and upper bounds of each variable are in the range of -1 and 1.
        super().__init__(n_var=self.n_vars, n_obj=2, xl=-1.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Method to evaluate the individual (x).
        Called by the optimization algorithm.
        Profit and drawdown are calculated based on the trading decisions agent makes in environment.
        The objectives are set to the profit and the negative drawdown.
        """
        # _evaluate is called from minimize() in main.py
        # "x" is an individial from the pop_size population
        # each individual is randomly initialized by NSGA-II's FloatRandomSampling()
        # each individual "x" has 6315 genes (the weights and bias values from the nerual network)
        # NSGA-II does mutation and crossover on highest performing individuals and
        # that is how genes (neural network weights and bias values) get changed (how learning happens)?
        self._decode_model(x)
        # print("type(x)", type(x))
        # print("x:", x)
        profit, drawdown = self.environment.simulate_trading()
        # print("profit:", profit, "drawdown:", drawdown, "num_trades:", num_trades)
        out["F"] = np.array([-profit, drawdown,])

    def _decode_model(self, params):
        """
        The most important method in this class.
        Decodes (i.e. maps) the genes of an individual (x) into the policy network.
        *** When stop-loss added we'll need to pop the last gene and return it to set the stop-loss value in the environment ***
        """
        idx = 0
        new_state_dict = {}

        # Adjust model attribute access
        model = self.network.module if isinstance(self.network, DataParallel) else self.network

        for name, param in model.named_parameters():
            num_param = param.numel()
            param_values = params[idx:idx + num_param]
            param_values = param_values.reshape(param.size())
            param_values = Tensor(param_values)
            new_state_dict[name] = param_values
            idx += num_param

        model.load_state_dict(new_state_dict)


class PerformanceLogger(Callback):
    def __init__(self, queue):
        super().__init__()
        self.history = []
        self.queue = queue

    def notify(self, algorithm):
        F = algorithm.pop.get("F")  # The objective values #algorithm.pop.non_dominated_inds
        X = algorithm.pop.get("X")  # The decision variables

        # Log the objective values (and any additional information)
        self.history.append({
            "generation": algorithm.n_gen,
            "objectives": F.copy(),
            "decision_variables": X.copy(),
        })
        # Add objective data to queue for plotting
        self.queue.put(self.history[-1]["objectives"])
