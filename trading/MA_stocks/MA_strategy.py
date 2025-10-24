import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import datetime
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class MovingAverageTradingStrategy:
    def __init__(self, symbol='AAPL', initial_capital=10000, hold_days=10):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.hold_days = hold_days  # Number of days to hold after buy signal
        self.api_key = os.getenv("API_KEY")
        self.secret_key = os.getenv("SECRET_KEY")
        self.client = StockHistoricalDataClient(api_key=self.api_key, secret_key=self.secret_key)
        
    def fetch_data(self, start_date, end_date):
        """Fetch stock data from Alpaca API"""
        request_params = StockBarsRequest(
            symbol_or_symbols=self.symbol,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )
        
        bars = self.client.get_stock_bars(request_params)
        data = bars.df
        
        # Clean the data
        data_clean = data.reset_index()
        data_clean = data_clean.set_index('timestamp')
        data_clean.index = pd.to_datetime(data_clean.index.date)
        data_clean = data_clean.drop('symbol', axis=1)
        
        return data_clean
    
    def calculate_moving_averages(self, df):
        """Calculate 20-day moving average"""
        df = df.copy()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        return df
    
    def generate_signals(self, df):
        """Generate buy/sell signals based on the strategy"""
        df = df.copy()
        
        # Buy signal: price moves above 20-day MA
        df['price_above_ma'] = df['close'] > df['ma_20']
        df['prev_price_above_ma'] = df['price_above_ma'].shift(1)
        
        # Buy when price crosses above MA (was below, now above)
        buy_condition = (df['price_above_ma'] == True) & (df['prev_price_above_ma'] == False)
        
        # Initialize signal columns
        df['signal'] = 0
        df['position'] = 0
        df['days_held'] = 0
        df['sell_date'] = pd.NaT
        df['bought_price'] = 0
        
        current_position = 0
        days_held = 0
        sell_date = None
        
        for i in range(len(df)):
            current_date = df.index[i]
            
            # Check for buy signal
            if buy_condition.iloc[i] and current_position == 0:
                df.iloc[i, df.columns.get_loc('signal')] = 1  # Buy signal
                current_position = 1
                days_held = 0
                # Set sell date to hold_days from now
                sell_date = current_date + pd.Timedelta(days=self.hold_days)
                df.iloc[i, df.columns.get_loc('sell_date')] = sell_date
            

            # Check for sell signal (after holding for specified days)
            elif current_position == 1 and sell_date is not None:
                days_held += 1
                if current_date >= sell_date:
                    df.iloc[i, df.columns.get_loc('signal')] = -1  # Sell signal
                    current_position = 0
                    days_held = 0
                    sell_date = None
            
            df.iloc[i, df.columns.get_loc('position')] = current_position
            df.iloc[i, df.columns.get_loc('days_held')] = days_held
        
        return df
    
    def backtest_strategy(self, df):
        """Backtest the trading strategy"""
        df = df.copy()
        
        # Initialize portfolio values
        df['portfolio_value'] = self.initial_capital
        df['shares'] = 0
        df['cash'] = self.initial_capital
        
        shares = 0
        cash = self.initial_capital
        
        for i in range(len(df)):
            if df['signal'].iloc[i] == 1:  # Buy signal
                # Buy as many shares as possible with available cash
                shares_to_buy = int(cash // df['close'].iloc[i])
                if shares_to_buy > 0:
                    shares += shares_to_buy
                    cash -= shares_to_buy * df['close'].iloc[i]
                    sell_date = df['sell_date'].iloc[i]
                    print(f"BUY: {df.index[i].date()} - Bought {shares_to_buy} shares at ${df['close'].iloc[i]:.2f} (Hold until {sell_date.date()})")
            
            elif df['signal'].iloc[i] == -1:  # Sell signal
                # Sell all shares
                if shares > 0:
                    cash += shares * df['close'].iloc[i]
                    print(f"SELL: {df.index[i].date()} - Sold {shares} shares at ${df['close'].iloc[i]:.2f} (Held for {self.hold_days} days)")
                    shares = 0
            
            # Update portfolio tracking
            df.iloc[i, df.columns.get_loc('shares')] = shares
            df.iloc[i, df.columns.get_loc('cash')] = cash
            df.iloc[i, df.columns.get_loc('portfolio_value')] = cash + shares * df['close'].iloc[i]
        
        return df
    
    def calculate_performance(self, df):
        """Calculate strategy performance metrics"""
        final_value = df['portfolio_value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        # Buy and hold return for comparison
        buy_hold_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
        
        # Calculate number of trades
        trades = len(df[df['signal'] != 0])
        
        print(f"\n=== PERFORMANCE SUMMARY ===")
        print(f"Symbol: {self.symbol}")
        print(f"Hold Period: {self.hold_days} days")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Portfolio Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
        print(f"Number of Trades: {trades}")
        print(f"Strategy vs Buy & Hold: {total_return - buy_hold_return:.2f}% difference")
        
        return {
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'final_value': final_value,
            'trades': trades
        }
    
    def plot_results(self, df):
        """Plot the results"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Price and moving averages with signals
        ax1.plot(df.index, df['close'], label='Close Price', linewidth=1)
        ax1.plot(df.index, df['ma_20'], label='20-day MA', alpha=0.7)
        
        # Mark buy and sell signals
        buy_signals = df[df['signal'] == 1]
        sell_signals = df[df['signal'] == -1]
        
        ax1.scatter(buy_signals.index, buy_signals['close'], color='green', marker='^', 
                   s=100, label='Buy Signal', zorder=5)
        ax1.scatter(sell_signals.index, sell_signals['close'], color='red', marker='v', 
                   s=100, label='Sell Signal', zorder=5)
        
        ax1.set_title(f'{self.symbol} - Hold for {self.hold_days} Days After MA Crossover')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Portfolio value over time
        ax2.plot(df.index, df['portfolio_value'], label='Strategy Portfolio', linewidth=2)
        
        # Calculate buy and hold portfolio for comparison
        buy_hold_portfolio = self.initial_capital * (df['close'] / df['close'].iloc[0])
        ax2.plot(df.index, buy_hold_portfolio, label='Buy & Hold', linewidth=2, alpha=0.7)
        
        ax2.set_title('Portfolio Performance Comparison')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.symbol}_strategy_results.png')
    
    def run_strategy(self, start_date, end_date):
        """Run the complete trading strategy"""
        print(f"Running Moving Average Crossover Strategy for {self.symbol}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Strategy: Buy when price moves above 20-day MA, hold for {self.hold_days} days")
        print("="*60)
        
        # Fetch and prepare data
        df = self.fetch_data(start_date, end_date)
        df = self.calculate_moving_averages(df)
        df = self.generate_signals(df)
        df = self.backtest_strategy(df)
        
        # Calculate and display performance
        performance = self.calculate_performance(df)
        
        # Plot results
        self.plot_results(df)
        
        return df, performance

# Example usage
if __name__ == "__main__":
    
    # Define date range
    start_date = datetime.datetime(2017, 1, 1)
    end_date = datetime.datetime(2025, 6, 30)
    
    results_df, performance = MovingAverageTradingStrategy(symbol='CVX', initial_capital=10000, hold_days=20).run_strategy(start_date, end_date)

    print("\n=== STRATEGY EXECUTION COMPLETE ===")

    results_df.to_csv('strategy_results.csv', index=True)

    stock_sharpe = results_df['close'].pct_change().mean() / results_df['close'].pct_change().std()
    print(f"Sharpe Ratio: {stock_sharpe:.2f}")
    
    portfolio_sharpe = results_df['portfolio_value'].pct_change().mean() / results_df['portfolio_value'].pct_change().std()
    print(f"Portfolio Sharpe Ratio: {portfolio_sharpe:.2f}")