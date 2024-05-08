import numpy as np
from torch import Tensor
from torch.nn import DataParallel
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.callback import Callback
from trading_environment import TradingEnvironment
from policy_network import PolicyNetwork


class TradingProblem(ElementwiseProblem):
    """
    The trading problem class for the multi-objective optimization.
    Takes a dataset, a policy network, and a trading environment.
    Calculates the number of variables based on the policy network's parameters.
    Calls the superclass constructor with the number of variables, objectives, and lower/upper bounds for x (gene) values.

    *** Still need to add 1 to n_vars for stop-loss gene ***
    """

    def __init__(self, data: Tensor, network: DataParallel | PolicyNetwork, environment: TradingEnvironment, *args, **kwargs):
        self.data = data
        self.network = network
        self.environment = environment

        # Adjust access to dims based on whether network is wrapped by DataParallel
        if isinstance(network, DataParallel):
            network_dims = network.module.dims
        else:
            network_dims = network.dims

        # Calculating the number of variables
        self.n_vars = sum([(network_dims[i] + 1) * network_dims[i + 1] for i in range(len(network_dims) - 1)])
        print("num vars", self.n_vars)
        # xl=-1.0, xu=1.0 is telling pymoo the lower and upper bounds of each variable are in the range of -1 and 1.
        # this is not true because of the way previous team
        super().__init__(n_var=self.n_vars, n_obj=3, xl=-1.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Method to evaluate the individual (x).
        Called by the optimization algorithm.
        Profit and drawdown are calculated based on the trading decisions agent makes in environment.
        The objectives are set to the profit and the negative drawdown.
        """
        self._decode_model(x)
        # print("type(x)", type(x))
        # print("x:", x)
        profit, drawdown, num_trades = self.environment.simulate_trading()
        out["F"] = np.array([-profit, drawdown, -num_trades])

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
        F = algorithm.pop.get("F")  # The objective values
        X = algorithm.pop.get("X")  # The decision variables
        # Log the objective values (and any additional information)
        self.history.append({
            "generation": algorithm.n_gen,
            "objectives": F.copy(),
            "decision_variables": X.copy(),
            "best": F.min(),
        })
        # Add objective data to queue for plotting
        self.queue.put(self.history[-1]["objectives"])
