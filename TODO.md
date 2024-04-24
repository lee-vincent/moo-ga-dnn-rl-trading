# TODO

## Eliza
- consolidate Instructions/installation_and_instructions.pdf into the README

## Jack

## Matt

## Michelle

## Vinnie
it is possible to compile pymoo modules: https://pymoo.org/installation.html
try a v100 on ubuntu python3.10, try compiling 3.12 on ubuntu, then try lowering code base to python 3.8 for running on rabbit
drop un-needed columns OHLV - and change close to adjclose
Current Data Date Ranges
stock data range        2011-01-01 -> 2023-12-31
training_tensor range   2011-01-01 -> 2022-01-01
testing_tensor range    2022-01-02 -> 2023-12-31

Proposed Data Date Ranges
stock data range        2011-01-01 -> DATE_PREVIOUS_MARKET_CLOSE
calculating DATE_PREVIOUS_MARKET_CLOSE:
DAYS_STOCK_MARKET_OPEN=[Monday,Tuesday,...,Friday]
TODAY is Monday 4-22-2024 
if TODAY in DAYS_STOCK_MARKET_OPEN
then DATE_PREVIOUS_MARKET_CLOSE = (TODAY - 1) = Friday 4-19-2024
training_tensor range     2011-01-01 - 2022-01-01
testing_tensor range    2022-01-02 - DATE_PREVIOUS_MARKET_CLOSE
DAY_TO_RUN_INFERENCE_ON = TODAY

always test on the last 365 days, and everything before going back to 2011 is training

spent time re-writing the yahoo_fin_data module to be ticker agnostic
mention all of the libraries/modules in main.py that we all had to learn about
spending significant time figuring out how to install python 3.12.3 on amazon linux 2
researching nvidia driver install
researching best AWS instance types that support GPUs for this use case
researching best type of nvidia gpu to use for this problem
had to resive EBS volumes in AWS many times to get it right
NVIDIA GRID K520
NVIDIA Tesla M60
NVIDIA T4 Tensor Core
NVIDIA A10G
NVIDIA Tesla K80
NVIDIA Tesla V100

might make sense to install NVIDIA Nsight for CUDA utilization monitoring

AWS has a free tier - could i get everyone set up in a free tier acount?

does compilied python modules make a speed difference
https://oregonstateuniversity-my.sharepoint.com/:p:/g/personal/leevi_oregonstate_edu/Ef0DT3pwcJVOoJtWvW97j1ABBXIVHZdChkwsiDDfygoDKg?e=ZSCJTz

pip3 install jupyter
pip3 install torch_tb_profiler
could use terraform to provision instances or work with the aws-algorithmic-trading repo


## data normalization
Example Normalization Process
Given your dataset with close_price, volume, price_velocity, and price_acceleration, here’s how you could apply StandardScaler:
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Example DataFrame
data = {
    'close_price': [120, 130, 125, 128, 132],
    'volume': [1500, 1600, 1580, 1620, 1700],
    'price_velocity': [-0.1, 0.2, 0.1, -0.2, 0.15],
    'price_acceleration': [-0.05, 0.03, 0.02, -0.01, 0.04]
}
df = pd.DataFrame(data)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the data
scaled_features = scaler.fit_transform(df)

# Create a new DataFrame with the scaled features
scaled_df = pd.DataFrame(scaled_features, columns=df.columns)

print(scaled_df)

```


## deployed ubuntu 22.04 LTS with 64GB EBS
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#system-requirements
follow the ubuntu 22.04 requirements
Ubuntu 22.04.z (z <= 3) LTS
kernel: 6.2.0-26
GCC 11.4.0
```bash
ssh -i ~/.ssh/bastion ubuntu@35.174.208.26
sudo apt update && sudo apt upgrade -y
sudo apt install gcc python3-pip libgl1 python-is-python3 -y
sudo reboot -f # because of kernel update message
# uname -r # 6.5.0-1017-aws
# python3 --version # Python 3.10.12
# pip3 --version # pip 22.0.2 from /usr/lib/python3/dist-packages/pip (python 3.10)
# Download CUDA Toolkit 12.4 Update 1 Installer for Linux Ubuntu 22.04 x86_64
# https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
sudo apt-get install -y cuda-drivers
export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}
sudo /usr/bin/nvidia-persistenced --verbose
# cat /proc/driver/nvidia/version
#   NVRM version: NVIDIA UNIX x86_64 Kernel Module  550.54.15  Tue Mar  5 22:23:56 UTC 2024
#   GCC version:  gcc version 11.4.0 (Ubuntu 11.4.0-1ubuntu1~22.04) 
# verify the installation
git config --global user.name "Vincent Lee"
git config --global user.email vinnie@vinnielee.io
git config --global credential.helper store
mkdir ~/repos
cd ~/repos/
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples/Samples/1_Utilities/deviceQuery
make
./deviceQuery
#  CUDA Device Query (Runtime API) version (CUDART static linking)

# Detected 1 CUDA Capable device(s)

# Device 0: "Tesla V100-SXM2-16GB"
#   CUDA Driver Version / Runtime Version          12.4 / 12.4
#   CUDA Capability Major/Minor version number:    7.0
#   Total amount of global memory:                 16144 MBytes (16928342016 bytes)
#   (080) Multiprocessors, (064) CUDA Cores/MP:    5120 CUDA Cores
#   GPU Max Clock rate:                            1530 MHz (1.53 GHz)
#   Memory Clock rate:                             877 Mhz
#   Memory Bus Width:                              4096-bit
#   L2 Cache Size:                                 6291456 bytes
#   Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
#   Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
#   Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
#   Total amount of constant memory:               65536 bytes
#   Total amount of shared memory per block:       49152 bytes
#   Total shared memory per multiprocessor:        98304 bytes
#   Total number of registers available per block: 65536
#   Warp size:                                     32
#   Maximum number of threads per multiprocessor:  2048
#   Maximum number of threads per block:           1024
#   Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
#   Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
#   Maximum memory pitch:                          2147483647 bytes
#   Texture alignment:                             512 bytes
#   Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
#   Run time limit on kernels:                     No
#   Integrated GPU sharing Host Memory:            No
#   Support host page-locked memory mapping:       Yes
#   Alignment requirement for Surfaces:            Yes
#   Device has ECC support:                        Enabled
#   Device supports Unified Addressing (UVA):      Yes
#   Device supports Managed Memory:                Yes
#   Device supports Compute Preemption:            Yes
#   Supports Cooperative Kernel Launch:            Yes
#   Supports MultiDevice Co-op Kernel Launch:      Yes
#   Device PCI Domain ID / Bus ID / location ID:   0 / 0 / 30
#   Compute Mode:
#      < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

# deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 12.4, CUDA Runtime Version = 12.4, NumDevs = 1
# Result = PASS
cd ../bandwidthTest/
make
./bandwidthTest
# [CUDA Bandwidth Test] - Starting...
# Running on...

#  Device 0: Tesla V100-SXM2-16GB
#  Quick Mode

#  Host to Device Bandwidth, 1 Device(s)
#  PINNED Memory Transfers
#    Transfer Size (Bytes)        Bandwidth(GB/s)
#    32000000                     11.1

#  Device to Host Bandwidth, 1 Device(s)
#  PINNED Memory Transfers
#    Transfer Size (Bytes)        Bandwidth(GB/s)
#    32000000                     12.5

#  Device to Device Bandwidth, 1 Device(s)
#  PINNED Memory Transfers
#    Transfer Size (Bytes)        Bandwidth(GB/s)
#    32000000                     739.1

# Result = PASS

# NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.
cd ~/repos/
git clone https://github.com/chesterornes/RL-Trading.git
cd ./RL-Trading/
git checkout lee-dev
export PATH=/home/ubuntu/.local/bin${PATH:+:${PATH}}
pip3 install -r requirements.txt 
python ./main.py
```

```bash
ubuntu@ip-172-31-43-83:~/repos/RL-Trading$ python ./main.py 
Population size: 100
Number of generations: 100
Profit threshold: 100.0
Drawdown threshold: 40.0

Continuing without tkinter backend...

split_index 2022-01-01 00:00:00 type <class 'pandas._libs.tslibs.timestamps.Timestamp'>
input_shape (main) 59
4 GPUs available.
4 GPUs available. Using DataParallel.
Process Process-1:
Traceback (most recent call last):
  File "/usr/lib/python3.10/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "/usr/lib/python3.10/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/ubuntu/repos/RL-Trading/./main.py", line 107, in train_and_validate
    problem = TradingProblem(data_collector.training_tensor, network,
  File "/home/ubuntu/repos/RL-Trading/trading_problem.py", line 24, in __init__
    self.n_vars = sum([(self.network.dims[i] + 1) * self.network.dims[i + 1] for i in range(len(self.network.dims) - 1)]) # The number of variables
  File "/home/ubuntu/.local/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1688, in __getattr__
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
AttributeError: 'DataParallel' object has no attribute 'dims'
^CTraceback (most recent call last):
  File "/home/ubuntu/repos/RL-Trading/./main.py", line 229, in <module>
    plotter.update_while_training()
  File "/home/ubuntu/repos/RL-Trading/plotter.py", line 185, in update_while_training
    time.sleep(0.1)
KeyboardInterrupt
```


automation workflow
aws resource creation - quota vcpu increases
testing automated update and configuration of project on instances
watched hours of videos https://www.youtube.com/watch?v=9Y3yaoi9rUQ


p3.8xlarge $12.2400 hourly $4.9759 hourly

g3.8xlarge	244.0 GiB	32 vCPUs	2	NVIDIA Tesla M60	
requires different nvidia drivers 
Data Center Driver For Ubuntu 22.04
https://www.nvidia.com/download/driverResults.aspx/210922/en-us/

or use pre-installed drivers on Amazon Linux 2 AMI ID: ami-0a8b4201c73c1b68f
https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html

for amazon linux 2, need to install python 3.12
https://techviewleo.com/how-to-install-python-on-amazon-linux-2/

Amazon Linux 2 P3
needs 64GB EBS volume

```bash
sudo yum update -y
sudo yum groupinstall "Development Tools" -y
sudo yum erase openssl-devel -y
sudo yum install openssl11 openssl11-devel  libffi-devel bzip2-devel wget tk-devel ncurses-devel uuid-devel.x86_64 readline-devel gdbm-devel xz-devel python3-tkinter.x86_64 -y
wget https://www.python.org/ftp/python/3.12.3/Python-3.12.3.tgz
tar -xf Python-3.12.3.tgz
cd Python-3.12.3/
./configure --enable-optimizations --prefix=/usr/local --enable-shared LDFLAGS="-Wl,-rpath /usr/local/lib"
make -j $(nproc)
sudo make altinstall
sudo yum install python3-pip -y
cd ~
git config --global user.name "Vincent Lee"
git config --global user.email vinnie@vinnielee.io
git config --global credential.helper store
mkdir ~/repos
cd ~/repos/
git clone https://github.com/chesterornes/RL-Trading.git
cd ./RL-Trading/
pip3.12 install -r requirements.txt 
python3.12 ./main.py

# got nvidia driver too old error so have to remove all nvidia and cuda drivers and re-install
# remove:
# install https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=CentOS&target_version=7&target_type=runfile_local
sudo yum -y erase nvidia cuda


sudo yum remove "cuda*" "*cublas*" "*cufft*" "*cufile*" "*curand*" \
 "*cusolver*" "*cusparse*" "*gds-tools*" "*npp*" "*nvjpeg*" "nsight*" "*nvvm*"
sudo yum remove "*nvidia*"
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
chmod +x cuda_12.4.1_550.54.15_linux.run
sudo sh cuda_12.4.1_550.54.15_linux.run
# have to enter 'accept'

```
# Previous Team's Work in Progress

## Data collection
1. Collect historical data for TQQQ
  1. Choose API
  2. Ensure TQQQ for time-period is available
  3. Decide which value(s) to focus on (open, high, low, close) **Most likely close
  4. Determine the structure for the data so data handling is clear
2. Clean the data for missing data or outliers
  1. Remove or account for missing data points
  2. Check for duplicate values
  3. ID outliers and decide to drop or transform the point 
3. Data visualization
  1. Evaluate raw data for any trends or other useful facts
  2. Determine size of regimes, etc.
4. Divide the data into training (2011-2021) and testing (2022-2023)

## Feature Engineering
1. Calculate RSI/SMA/ADX (other measures/indicators?) values to trigger the buy/sell decisions
2. Explore other features/indices
3. Normalize and/or scale features as needed
  1. Z-score normalization
  2. Min-Max scaling

## Architecture
1. **Neural network(s)**
  1. Design neural network to predict buy/sell based on indices
    1. Input features from above
    2. Design output layer with 3 neurons, 1 for each possible decision
    3. Activation function???
      1. Softmax function might be an option
        1. Example code for `tensorflow` below
    4. Determine loss function - cross-entropy assuming the signal should be a classification problem
  2. Train the NN with the training data
    1. Experiment with number of hidden layers and number of neurons
    2. Choose activation function for the hidden layers (tanh/ReLU?)
  3. Determine the threshold for decision
    1. High-value: buy, middle-range: hold, low-value: sell
    2. Example: value >= 0.65 -> buy
  4. Use LSTMs or RNNs? (Thought we're not using these anymore?)
    1. RNN is simpler than LSTMs while LSTMs work great for time-sequence data (like stocks)
    2. Would require transforming the input data into windows of time sequences
    3. Reshape input into 3D array
      1. `(num_records, num_windows, num_features)`
    4. See below for alternative code using LSTM 
2. **Genetic algorithm**
  1. Optimize for low drawdown and high profitability
  2. Define chromosomes
    1. Based on features, risk management plan
    2. Convert NN prediction into decisions based on the thresholds
  3. Define a function that creates a fitness score
  4. Create a starting population
  5. Run genetic algorithm
  6. Find the Pareto frontier
  7. Allow for selecting which point along the frontier is desired by the user based on risk

## Evaluate
1. Decide on performance metrics
2. Test model on training data
  1. Check for overfitting, other issues
  2. Study differences in baseline algorithm trading strategy and model produced data
3. Test model on test data
  1. Accuracy
  2. Profitability
  3. Max draw down
4. Iteratively work on model architecture

## Risk Management
1. Define as a stop-loss value (number or percent) 
  1. Or by the chosen optimization model from the genetic algorithm?
  2. What is the max acceptable loss allowed before selling?
  3. Constantly monitor checking for stop-loss trigger?
2. Define a take-profit threshold to ensure profit?

## Deploy trading agent?
1. Connect to trading API
2. Monitor performance
3. Retrain with more recent data (months or years later)?

## Documentation
1. Overview
2. Data collection process
3. Preprocessing
4. Model architecture
5. Training process
6. Hyperparameter tuning
7. Integration with genetic algorithm
8. Risk Management
9. Results and Performance
10. Projected next steps
11. Code
  1. Dependencies and environment
12. Challenges
13. Acknowledgements
14. References
15. Legal disclaimer?


## Example Code
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

# num_input_features would be the vector(array) of feature (close, SMA, RSI, ADX, etc.) and date data

# Model
model = Sequential()  # initialize an empty sequential model
model.add(Dense(units=hidden_units, input_dim=num_input_features, activation='relu))  # hidden layer using ReLU for the activation function
model.add(Dense(units=3))  # Output layer has 3 neurons (buy/sell/hold)
model.add(Activation('softmax'))  # Used for multi-class classification


# Alternative using LSTM instead of ReLU activation function
from tensorflow.keras.layers import LSTM

# Variables needed for windows and the number of features
num_windows = ?
num_features = ?

# Transform input data
input_reshaped = input.reshape((input.shape[0], num_windows, num_features))

model = Sequential()  # initialize an empty sequential model
model.add(LSTM(units=50, input_shape(num_windows, num_features), return_sequences=True))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(units=3, activation='softmax'))
```
