# Limit Order Execution Environment

This project implements a reinforcement learning environment for limit order execution in financial markets using the Gymnasium framework and Stable Baselines3.

## Description

The LimitOrderEnv simulates a trading scenario where an agent must execute a large order over a specified time horizon. The agent decides on the limit price for each order, balancing between aggressive execution and minimizing market impact.

## Features

- Custom Gymnasium environment for limit order execution
- Integration with Stable Baselines3 for easy training and testing of RL algorithms
- Configurable parameters such as initial shares, time horizon, and tick size
- Command-line interface for training and testing modes

## Requirements
- gymnasium
- stable-baselines3

You can install the required packages using:

```
pip install -r requirements.txt
```

### Training

To train a new model:

```
python main.py --mode train --model_dir models
```

This will start training a PPO model and save checkpoints in the specified `model_dir`.

### Testing

To test a trained model:

```
python main.py --mode test --model_path models/10000
```

Replace `models/10000` with the path to your trained model.

## Environment Details

- State space: 6-dimensional vector including current step, remaining shares, and top-of-book market data
- Action space: Continuous action in [-1, 1] representing the price deviation from the current ask price
- Reward: Negative implementation shortfall (difference between execution price and initial mid-price)

## Customization

You can modify the following parameters in the `LimitOrderEnv` class:

- `csv_path`: Path to the market data CSV file
- `initial_shares`: Number of shares to execute
- `time_horizon`: Number of time steps for execution
- `tick_size`: Minimum price movement
