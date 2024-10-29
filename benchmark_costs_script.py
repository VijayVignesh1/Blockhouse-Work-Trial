import pandas as pd
import numpy as np

class Benchmark:
    
    def __init__(self, data):
        """
        Initializes the Benchmark class with provided market data.

        Parameters:
        data (DataFrame): A DataFrame containing market data, including top 5 bid prices and sizes. (Use bid_ask_ohlcv_data)
        """
        # Use data with top 5 bid prices and sizes for benchmarking
        self.data = data

    def get_twap_trades(self, data, initial_inventory, preferred_timeframe=390):
        """
        Generates a trade schedule based on the Time-Weighted Average Price (TWAP) strategy.

        Parameters:
        data (DataFrame): The input data containing timestamps and closing prices for each time step.
        initial_inventory (int): The total number of shares to be sold over the preferred timeframe.
        preferred_timeframe (int): The total number of time steps (default is 390, representing a full trading day).

        Returns:
        DataFrame: A DataFrame containing the TWAP trades with timestamps, price, shares sold, and remaining inventory.
        """
        total_steps = len(data)
        twap_shares_per_step = initial_inventory / preferred_timeframe
        remaining_inventory = initial_inventory
        trades = []
        for step in range(min(total_steps, preferred_timeframe)):
            size_of_slice = min(twap_shares_per_step, remaining_inventory)
            remaining_inventory -= int(np.ceil(size_of_slice))
            trade = {
                'timestamp': data.iloc[step]['timestamp'],
                'step': step,
                'price': data.iloc[step]['bid_price_1'],
                'shares': size_of_slice,
                'inventory': remaining_inventory,
            }
            trades.append(trade)
        return pd.DataFrame(trades)

    def get_vwap_trades(self, data, initial_inventory, preferred_timeframe=390):
        """
        Generates a trade schedule based on the Volume-Weighted Average Price (VWAP) strategy.

        Parameters:
        data (DataFrame): The input data containing timestamps, closing prices, and volumes for each time step.
        initial_inventory (int): The total number of shares to be sold over the preferred timeframe.
        preferred_timeframe (int): The total number of time steps (default is 390, representing a full trading day).

        Returns:
        DataFrame: A DataFrame containing the VWAP trades with timestamps, price, shares sold, and remaining inventory.
        """
        total_volume = data['bid_size_1'].sum()
        total_steps = len(data)
        remaining_inventory = initial_inventory
        trades = []
        for step in range(min(total_steps, preferred_timeframe)):
            volume_at_step = data['bid_size_1'].iloc[step]
            size_of_slice = (volume_at_step / total_volume) * initial_inventory
            size_of_slice = min(size_of_slice, remaining_inventory)
            remaining_inventory -= int(np.ceil(size_of_slice))
            trade = {
                'timestamp': data.iloc[step]['timestamp'],
                'step': step,
                'price': data.iloc[step]['bid_price_1'],
                'shares': size_of_slice,
                'inventory': remaining_inventory,
            }
            trades.append(trade)
        return pd.DataFrame(trades)
    
    def calculate_vwap(self, idx, shares):
        """
        Calculates the Volume-Weighted Average Price (VWAP) for a given step and share size.

        Parameters:
        idx (int): The index of the current step in the market data.
        shares (int): The number of shares being traded at the current step.

        Returns:
        float: The calculated VWAP price for the current step.
        """
        # Assumes you have best 5 bid prices and sizes in your dataset
        bid_prices = np.array([self.data.iloc[idx][f'bid_price_{i}'] for i in range(1,6)])
        bid_sizes = np.array([self.data.iloc[idx][f'bid_size_{i}'] for i in range(1,6)])
        cumsum = 0
        for idx, size in enumerate(bid_sizes):
            cumsum += size
            if cumsum >= shares:
                break
        
        return np.sum(bid_prices[:idx+1] * bid_sizes[:idx+1]) / np.sum(bid_sizes[:idx+1])

    def calculate_twap(self, start_idx, end_idx):
        """
        Calculates the Time-Weighted Average Price (TWAP) for a given time range.

        Parameters:
        start_idx (int): The starting index of the time range in the market data.
        end_idx (int): The ending index of the time range in the market data.

        Returns:
        float: The calculated TWAP price for the given time range.
        """
        # Calculate mid-prices for the time range
        mid_prices = np.array([
            (self.data.iloc[i]['bid_price_1'] + self.data.iloc[i]['ask_price_1']) / 2
            for i in range(start_idx, end_idx + 1)
        ])
        
        # TWAP is simply the average of mid-prices over time
        return np.mean(mid_prices)

    def compute_components(self, alpha, shares, actual_price, idx):
        """
        Computes the transaction cost components such as slippage and market impact for a given trade.

        Parameters:
        alpha (float): A scaling factor for market impact (determined empirically or based on research).
        shares (int): The number of shares being traded at the current step.
        idx (int): The index of the current step in the market data.

        Returns:
        array: A NumPy array containing the slippage and market impact for the given trade.
        """
        Slippage = (self.data.iloc[idx]['bid_price_1'] - actual_price) * shares  # Assumes bid_price is in your dataset
        Market_Impact = alpha * np.sqrt(shares)
        return np.array([Slippage, Market_Impact])
    
    def simulate_strategy_vwap(self, trades, data, preferred_timeframe):
        """
        Simulates a trading strategy and calculates various transaction cost components.

        Parameters:
        trades (DataFrame): A DataFrame where each row contains 'shares' and 'action' for each trade.
        data (DataFrame): Market data including bid prices and volumes.
        preferred_timeframe (int): The total number of time steps over which the strategy is simulated.

        Returns:
        tuple: A tuple containing lists of slippage, market impact.
        """
        
        # Initialize result lists
        slippage = []
        market_impact = []
        alpha = 4.439584265535017e-06 
        rewards = []
        shares_traded = []

        # Simulate the strategy
        for idx in range(len(trades)):
            shares = trades.iloc[idx]['shares']
            actual_price = self.calculate_vwap(idx, shares)
            reward = self.compute_components(alpha, shares, actual_price, idx)
            slippage.append(reward[0])
            market_impact.append(reward[1])
            shares_traded.append(shares)
            rewards.append(reward)

        return slippage, market_impact

    def simulate_strategy_twap(self, trades, data, preferred_timeframe):
        """
        Simulates a trading strategy and calculates various transaction cost components.

        Parameters:
        trades (DataFrame): A DataFrame where each row contains 'shares' and 'action' for each trade.
        data (DataFrame): Market data including bid prices and volumes.
        preferred_timeframe (int): The total number of time steps over which the strategy is simulated.

        Returns:
        tuple: A tuple containing lists of slippage, market impact.
        """
        
        # Initialize result lists
        slippage = []
        market_impact = []
        alpha = 4.439584265535017e-06 
        rewards = []
        shares_traded = []

        # Simulate the strategy
        for idx in range(len(trades)):
            shares = trades.iloc[idx]['shares']
            actual_price = self.calculate_twap(max(0, idx - 10), idx)
            reward = self.compute_components(alpha, shares, actual_price, idx)
            slippage.append(reward[0])
            market_impact.append(reward[1])
            shares_traded.append(shares)
            rewards.append(reward)

        return slippage, market_impact

    def simulate_strategy(self, trades, data, preferred_timeframe):
        """
        Simulates a trading strategy and calculates various transaction cost components.

        Parameters:
        trades (DataFrame): A DataFrame where each row contains 'shares' and 'action' for each trade.
        data (DataFrame): Market data including bid prices and volumes.
        preferred_timeframe (int): The total number of time steps over which the strategy is simulated.

        Returns:
        tuple: A tuple containing lists of slippage, market impact.
        """
        
        # Initialize result lists
        slippage = []
        market_impact = []
        alpha = 4.439584265535017e-06 
        rewards = []
        shares_traded = []

        # Simulate the strategy
        for idx in range(len(trades)):
            shares = trades.iloc[idx]['shares']
            actual_price = trades.iloc[idx]['price']
            reward = self.compute_components(alpha, shares, actual_price, idx)
            slippage.append(reward[0])
            market_impact.append(reward[1])
            shares_traded.append(shares)
            rewards.append(reward)

        return slippage, market_impact