from torch import nn


class PolicyNetwork(nn.Module):
    """A simple feedforward neural network for policy approximation."""
    def __init__(self, activation_function, dimensions):
        super(PolicyNetwork, self).__init__()  # Call the superclass constructor
        self.dims = dimensions  # The dimensions of the network [54, 64, 32, 16, 8, 4, 3]
        self.network = nn.Sequential(
            # 54*64 weights + 64 biases
            nn.Linear(self.dims[0], self.dims[1]),
            nn.ReLU() if activation_function == 'ReLU' else nn.Tanh(),  # Conditional activation function
            # 64*32 weights + 32 biases
            nn.Linear(self.dims[1], self.dims[2]),
            nn.ReLU() if activation_function == 'ReLU' else nn.Tanh(),  # Conditional activation function
            # 32*16 weights + 16 biases
            nn.Linear(self.dims[2], self.dims[3]),
            nn.ReLU() if activation_function == 'ReLU' else nn.Tanh(),  # Conditional activation function
            # 16*8 weights + 8 biases
            nn.Linear(self.dims[3], self.dims[4]),
            nn.ReLU() if activation_function == 'ReLU' else nn.Tanh(),  # Conditional activation function
            # 8*4 weights + 4 biases
            nn.Linear(self.dims[4], self.dims[5]),
            nn.ReLU() if activation_function == 'ReLU' else nn.Tanh(),  # Conditional activation function
            # 4*3 weights + 3 biases
            nn.Linear(self.dims[5], self.dims[6]),
            nn.Softmax(dim=-1)
        )

        print(self.network[0].weight)
        print(self.network[0].bias)
        # exit()

    def forward(self, x):
        """Forward pass of the neural network."""
        return self.network(x)
