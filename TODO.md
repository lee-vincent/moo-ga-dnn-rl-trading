# TODO

## Eliza
- consolidate Instructions/installation_and_instructions.pdf into the README

## Jack

## Matt

## Michelle

## Vinnie
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

AWS has a free tier - could i get everyone set up in a free tier acount?

could use terraform to provision instances or work with the aws-algorithmic-trading repo


deployed ubuntu 22.04 LTS

ssh -i ~/.ssh/bastion ubuntu@35.174.208.26
python3 --version
Python 3.10.12
sudo apt update && sudo apt upgrade -y
or this one? sudo apt-get update && sudo apt-get upgrade -y
sudo apt install python3-pip -y
keeps asking for kernal upgrade??

sudo apt-get install python-is-python3
sudo apt-get install libgl1 -y
git config --global user.name "Vincent Lee"
git config --global user.email vinnie@vinnielee.io
git config --global credential.helper store


mkdir ~/repos
cd ~/repos/
git clone https://github.com/chesterornes/RL-Trading.git
cd ./RL-Trading/
pip3.12 install -r requirements.txt 
python3.12 ./main.py


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
