from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
import gymnasium as gym
import numpy as np
import pandas as pd

class LimitOrderEnv(gym.Env):
    def __init__(self, csv_path="AAPL_Quotes_Data.csv", initial_shares=1000, time_horizon=390, tick_size=0.1):
        super(LimitOrderEnv, self).__init__()
        
        self.data = pd.read_csv(csv_path)
        self.initial_shares = initial_shares
        self.time_horizon = time_horizon 
        self.tick_size = tick_size
        self.max_price_deviation = 15
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(6,), dtype=np.float32)
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.initial_mid_price = self._calculate_mid_price()
        self.remaining_shares = self.initial_shares
        self.current_step = 0
        self.total_value = 0
        self.reward = 0
        
        return self._get_observation(), {}

    def step(self, action):
        price_deviation = action[0] * self.max_price_deviation
        executed_shares = 0
        
        order_price = self.data.iloc[self.current_step]['ask_price_1'] - (price_deviation * self.tick_size)
        
        if self.current_step != self.time_horizon - 1:
            bid_price = self.data.iloc[self.current_step]['bid_price_1']
            bid_size = self.data.iloc[self.current_step]['bid_size_1']

            if bid_price >= order_price:
                executed = min(self.remaining_shares, bid_size)
                executed_shares += executed
                self.total_value += executed * bid_price

                self.remaining_shares -= executed_shares
        else:
            price_deviation = 0
            order_price = self.data.iloc[self.current_step]['ask_price_1'] - (price_deviation * self.tick_size)
            for i in range(1,6):
                bid_price = self.data.iloc[self.current_step][f'bid_price_{i}']
                bid_size = self.data.iloc[self.current_step][f'bid_size_{i}']

                if bid_price >= order_price:
                    executed = min(self.remaining_shares, bid_size)
                    executed_shares += executed
                    self.total_value += executed * bid_price

                    self.remaining_shares -= executed_shares
                
        self.current_step += 1
        done = self.remaining_shares <= 0

        reward = self._calculate_reward(executed_shares, order_price)
        self.render()
        return self._get_observation(), reward, done, False, {
            "executed_volume": executed_shares, 
            "order_price": order_price
        }

    def render(self):
        print(self.current_step, self.remaining_shares)

    def _calculate_mid_price(self):
        best_bid = self.data['bid_price_1'].iloc[0]
        best_ask = self.data['ask_price_1'].iloc[0]
        return (best_bid + best_ask) / 2

    def _calculate_reward(self, executed_volume, execution_price):

        cash_received = executed_volume * execution_price
        cost = -(cash_received - (self.initial_mid_price * executed_volume))
        reward = -cost
        return reward 
    
    def _get_observation(self):
        obs = [
            self.current_step,
            self.remaining_shares / self.initial_shares
        ]
        obs.append(self.data.iloc[self.current_step]['bid_price_1'])
        obs.append(self.data.iloc[self.current_step]['bid_size_1'])
        obs.append(self.data.iloc[self.current_step]['ask_price_1'])
        obs.append(self.data.iloc[self.current_step]['ask_size_1'])
        return np.array(obs, dtype=np.float32)