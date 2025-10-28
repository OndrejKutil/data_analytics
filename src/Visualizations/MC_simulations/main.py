import numpy as np
from dataclasses import dataclass
from typing import List
from scipy import stats


@dataclass
class GBMParams:
    """Parameters for Geometric Brownian Motion simulation"""
    S0: float        # Initial stock price
    mu: float        # Expected return (drift)
    sigma: float     # Volatility
    T: float         # Years to simulate
    steps: int       # Number of time steps
    paths: int       # Number of simulation paths


class GBMStockSimulation:
    def __init__(self, parameters: GBMParams):
        self.params = parameters
    
    def random_double(self, lower_bound: float, upper_bound: float) -> float:
        """Generate random number from normal distribution"""
        if lower_bound > upper_bound:
            raise ValueError("lower_bound must be less than or equal to upper_bound")
        if lower_bound == upper_bound:
            return lower_bound
        
        # Generate from normal distribution (mean=lower_bound, std based on range)
        return np.random.normal(lower_bound, upper_bound)
    
    def gbm_stocks(self, params: GBMParams) -> List[List[float]]:
        """
        Simulate stock prices using Geometric Brownian Motion
        
        Returns:
            List of price paths, where each path is a list of prices at each time step
        """
        prices = []
        
        S0 = params.S0
        mu = params.mu
        sigma = params.sigma
        T = params.T
        steps = params.steps
        paths = params.paths
        
        dt = T / steps
        drift = (mu - 0.5 * sigma * sigma) * dt
        diffusion_coeff = sigma * np.sqrt(dt)
        
        for p in range(paths):
            path_prices = []
            s = S0
            
            for t in range(steps):
                Z = self.random_double(0.0, 1.0)
                diffusion = diffusion_coeff * Z
                s = s * np.exp(drift + diffusion)
                path_prices.append(s)
            
            prices.append(path_prices)
        
        return prices


def main():
    years = 10
    trading_days_per_year = 252
    
    params = GBMParams(
        S0=100.0,       # Initial stock price
        mu=0.07,        # Drift (expected return)
        sigma=0.2,      # Volatility
        T=years,        # Years to simulate
        steps=years * trading_days_per_year,  # Total time steps across all years
        paths=1000,     # Number of simulation paths
    )
    
    simulator = GBMStockSimulation(params)
    prices = simulator.gbm_stocks(params)
    
    # plot the results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    for path in prices:
        plt.plot(path)
    plt.title("Geometric Brownian Motion - Stock Price Simulation")
    plt.xlabel("Time Steps")
    plt.ylabel("Stock Price")
    #plt.savefig("gbm_stock_simulation.png")

    final_prices = [path[-1] for path in prices]

    mean_final_price = np.mean(final_prices)
    median_final_price = np.median(final_prices)
    std_final_price = np.std(final_prices)

    # plot histogram of final prices
    # also plot the summary statistics: mean, median, mode, std and poisson distribution
    plt.figure(figsize=(10, 5))
    plt.hist(final_prices, bins=100, alpha=0.7)
    plt.title("Histogram of Final Prices")
    plt.xlabel("Final Price")
    plt.ylabel("Frequency")


    print("Summary Statistics of Final Prices:")
    print(f"Mean: {mean_final_price}")
    print(f"Median: {median_final_price}")
    print(f"Standard Deviation: {std_final_price}")

if __name__ == "__main__":
    main()
