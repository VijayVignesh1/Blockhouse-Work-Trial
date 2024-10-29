import gymnasium as gym
from gymnasium.envs.registration import register
import os
import argparse
from stable_baselines3 import PPO
from limit_order import LimitOrderEnv
from benchmark_costs_script import Benchmark
import pandas as pd

register(
    id='limit-order-v0',                                
    entry_point='limit_order:LimitOrderEnv', 
)

def train(env, model_dir, log_dir):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    model = PPO('MlpPolicy', env, verbose=1, device='cpu', tensorboard_log=log_dir)
   
    TIMESTEPS = 1000
    iters = 0
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{model_dir}/{TIMESTEPS*iters}")

def test(env, model_path):
    model = PPO.load(model_path, env=env)

    obs = env.reset()[0]
    terminated = False
    while not terminated:
        action, _ = model.predict(observation=obs, deterministic=True)
        obs, _, terminated, _, _ = env.step(action)


def backtest_ppo(model_path, data_path, initial_shares, time_horizon):
    # Load the data
    data = pd.read_csv(data_path)
    
    # Create the environment
    env = gym.make('limit-order-v0', csv_path=data_path, initial_shares=initial_shares, time_horizon=time_horizon)
    
    # Load the trained model
    model = PPO.load(model_path, env=env)
    
    # Initialize variables for PPO simulation
    obs, _ = env.reset()
    done = False
    policy_trades = []
    
    # Simulate PPO strategy
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        policy_trades.append({
            'timestamp': data.iloc[env.current_step-1]['timestamp'] if 'timestamp' in data.columns else env.current_step-1,
            'step': env.current_step-1,
            'price': info['order_price'],
            'shares': info['executed_volume'],
            'inventory': env.remaining_shares
        })
    
    policy_trades_df = pd.DataFrame(policy_trades)
    
    # Create Benchmark instance
    benchmark = Benchmark(data)
    vwap_trades = benchmark.get_vwap_trades(data, initial_shares, time_horizon)
    twap_trades = benchmark.get_twap_trades(data, initial_shares, time_horizon)

    twap_metrics = benchmark.simulate_strategy_twap(twap_trades, data, time_horizon) 
    vwap_metrics= benchmark.simulate_strategy_vwap(vwap_trades, data, time_horizon)
    policy_metrics = benchmark.simulate_strategy(policy_trades_df, data, time_horizon)
    
    return policy_metrics, vwap_metrics, twap_metrics

def main():
    parser = argparse.ArgumentParser(description="Train or test the LimitOrder environment")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help='Mode: train or test')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save/load models')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for tensorboard logs')
    parser.add_argument('--model_path', type=str, help='Path to the model for testing')
    parser.add_argument('--data_path', type=str, default="AAPL_Quotes_Data.csv", help='Path to the model for testing')
    args = parser.parse_args()

    env = gym.make('limit-order-v0')
    
    if args.mode == 'train':
        train(env, args.model_dir, args.log_dir)
    elif args.mode == 'test':
        if args.model_path is None:
            raise ValueError("Model path must be provided for testing")
        test(env, args.model_path)
        backtest_ppo(args.model_path, args.data_path, 1000, 390)

if __name__ == "__main__":
    main()
