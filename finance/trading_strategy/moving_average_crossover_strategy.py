import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class MovingAverageCrossoverStrategy:
    """
    A simple Moving Average Crossover backtesting strategy.
    Buy when 50-day MA crosses above 200-day MA (Golden Cross)
    Sell when 50-day MA crosses below 200-day MA (Death Cross)
    """
    
    def __init__(self, ticker, initial_capital=10000, short_window=50, long_window=200):
        self.ticker = ticker
        self.initial_capital = initial_capital
        self.short_window = short_window
        self.long_window = long_window
        self.data = None
        self.results = None
        
    def fetch_data(self, period='max'):
        """Download and prepare data with moving averages"""
        print(f"📈 Fetching data for {self.ticker}...")
        
        # Download data
        raw_data = yf.download(self.ticker, period=period, interval='1d', auto_adjust=True)
        
        # Handle the data structure (single ticker gives different format)
        if isinstance(raw_data.columns, pd.MultiIndex):
            # Multiple tickers case
            self.data = pd.DataFrame({
                'Close': raw_data['Close'][self.ticker],
                'Volume': raw_data['Volume'][self.ticker]
            }, index=raw_data.index)
        else:
            # Single ticker case
            self.data = pd.DataFrame({
                'Close': raw_data['Close'],
                'Volume': raw_data['Volume']
            }, index=raw_data.index)
        
        # Calculate moving averages
        self.data[f'{self.short_window}MA'] = self.data['Close'].rolling(window=self.short_window).mean()
        self.data[f'{self.long_window}MA'] = self.data['Close'].rolling(window=self.long_window).mean()
        
        # Remove NaN values
        self.data = self.data.dropna()
        
        print(f"✅ Data loaded: {len(self.data)} trading days from {self.data.index[0].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')}")
        
    def generate_signals(self):
        """Generate buy/sell signals based on MA crossover"""
        print("🔄 Generating trading signals...")
        
        # Detect crossovers
        self.data['Signal'] = 0
        self.data['Position'] = 0
        
        # Golden Cross: 50MA crosses above 200MA (Buy signal)
        golden_cross = (self.data[f'{self.short_window}MA'] > self.data[f'{self.long_window}MA']) & \
                      (self.data[f'{self.short_window}MA'].shift(1) <= self.data[f'{self.long_window}MA'].shift(1))
        
        # Death Cross: 50MA crosses below 200MA (Sell signal)
        death_cross = (self.data[f'{self.short_window}MA'] < self.data[f'{self.long_window}MA']) & \
                     (self.data[f'{self.short_window}MA'].shift(1) >= self.data[f'{self.long_window}MA'].shift(1))
        
        # Set signals
        self.data.loc[golden_cross, 'Signal'] = 1  # Buy
        self.data.loc[death_cross, 'Signal'] = -1  # Sell
        
        # Track position (1 = holding, 0 = not holding)
        self.data['Position'] = self.data['Signal'].replace(to_replace=0, method='ffill').fillna(0)
        self.data['Position'] = self.data['Position'].replace(-1, 0)  # Convert sell signals to no position
        
        # Mark actual trade entry/exit points
        self.data['Trade_Signal'] = self.data['Signal'].copy()
        
        buy_signals = len(self.data[self.data['Signal'] == 1])
        sell_signals = len(self.data[self.data['Signal'] == -1])
        
        print(f"📊 Generated {buy_signals} buy signals and {sell_signals} sell signals")
        
    def backtest_strategy(self):
        """Execute the backtesting logic"""
        print("⚙️ Running backtest...")
        
        # Initialize tracking variables
        cash = self.initial_capital
        shares = 0
        portfolio_value = []
        trade_log = []
        
        for date, row in self.data.iterrows():
            current_price = row['Close']
            signal = row['Signal']
            
            # Execute trades
            if signal == 1 and shares == 0:  # Buy signal and not holding
                shares = cash / current_price
                cash = 0
                trade_log.append({
                    'Date': date,
                    'Action': 'BUY',
                    'Price': current_price,
                    'Shares': shares,
                    'Portfolio_Value': shares * current_price
                })
                
            elif signal == -1 and shares > 0:  # Sell signal and holding
                cash = shares * current_price
                trade_log.append({
                    'Date': date,
                    'Action': 'SELL',
                    'Price': current_price,
                    'Shares': shares,
                    'Portfolio_Value': cash
                })
                shares = 0
            
            # Calculate current portfolio value
            if shares > 0:
                portfolio_value.append(shares * current_price)
            else:
                portfolio_value.append(cash)
        
        # Store results
        self.data['Portfolio_Value'] = portfolio_value
        self.trade_log = pd.DataFrame(trade_log)
        
        # Calculate final portfolio value
        final_value = portfolio_value[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        print(f"🎯 Backtest completed!")
        print(f"   Initial Capital: ${self.initial_capital:,.2f}")
        print(f"   Final Value: ${final_value:,.2f}")
        print(f"   Total Return: {total_return:.2f}%")
        print(f"   Total Trades: {len(self.trade_log)}")
        
    def calculate_performance_metrics(self):
        """Calculate detailed performance metrics"""
        print("\n📊 PERFORMANCE ANALYSIS")
        print("=" * 50)
        
        # Basic metrics
        final_value = self.data['Portfolio_Value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        # Buy and hold comparison
        buy_hold_return = (self.data['Close'].iloc[-1] - self.data['Close'].iloc[0]) / self.data['Close'].iloc[0] * 100
        
        # Calculate trade-level metrics
        trades = []
        for i in range(0, len(self.trade_log), 2):
            if i + 1 < len(self.trade_log):
                buy_trade = self.trade_log.iloc[i]
                sell_trade = self.trade_log.iloc[i + 1]
                trade_return = (sell_trade['Price'] - buy_trade['Price']) / buy_trade['Price'] * 100
                trades.append(trade_return)
        
        win_rate = len([t for t in trades if t > 0]) / len(trades) * 100 if trades else 0
        avg_return = np.mean(trades) if trades else 0
        
        # Maximum drawdown
        portfolio_peak = self.data['Portfolio_Value'].expanding().max()
        drawdown = (self.data['Portfolio_Value'] - portfolio_peak) / portfolio_peak * 100
        max_drawdown = drawdown.min()
        
        # Print results
        print(f"Strategy Return:     {total_return:.2f}%")
        print(f"Buy & Hold Return:   {buy_hold_return:.2f}%")
        print(f"Outperformance:      {total_return - buy_hold_return:.2f}%")
        print(f"Total Trades:        {len(self.trade_log)}")
        print(f"Completed Trades:    {len(trades)}")
        print(f"Win Rate:           {win_rate:.1f}%")
        print(f"Avg Trade Return:   {avg_return:.2f}%")
        print(f"Max Drawdown:       {max_drawdown:.2f}%")
        
        # Trading period
        start_date = self.data.index[0].strftime('%Y-%m-%d')
        end_date = self.data.index[-1].strftime('%Y-%m-%d')
        total_days = (self.data.index[-1] - self.data.index[0]).days
        print(f"Period:             {start_date} to {end_date} ({total_days} days)")
        
    def plot_results(self):
        """Create comprehensive visualization of the strategy"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Price and Moving Averages with Signals
        ax1.plot(self.data.index, self.data['Close'], label=f'{self.ticker} Price', color='black', linewidth=1)
        ax1.plot(self.data.index, self.data[f'{self.short_window}MA'], label=f'{self.short_window}-day MA', color='blue', alpha=0.7)
        ax1.plot(self.data.index, self.data[f'{self.long_window}MA'], label=f'{self.long_window}-day MA', color='red', alpha=0.7)
        
        # Mark buy/sell signals
        buy_signals = self.data[self.data['Signal'] == 1]
        sell_signals = self.data[self.data['Signal'] == -1]
        
        ax1.scatter(buy_signals.index, buy_signals['Close'], color='green', marker='^', s=100, label='Buy Signal', zorder=5)
        ax1.scatter(sell_signals.index, sell_signals['Close'], color='red', marker='v', s=100, label='Sell Signal', zorder=5)
        
        ax1.set_title(f'{self.ticker} Price with Moving Average Crossover Signals', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Portfolio Value Over Time
        ax2.plot(self.data.index, self.data['Portfolio_Value'], label='Strategy Portfolio', color='green', linewidth=2)
        
        # Buy and hold comparison
        buy_hold_value = self.initial_capital * (self.data['Close'] / self.data['Close'].iloc[0])
        ax2.plot(self.data.index, buy_hold_value, label='Buy & Hold', color='blue', linewidth=2, alpha=0.7)
        
        ax2.set_title('Portfolio Value Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Drawdown
        portfolio_peak = self.data['Portfolio_Value'].expanding().max()
        drawdown = (self.data['Portfolio_Value'] - portfolio_peak) / portfolio_peak * 100
        
        ax3.fill_between(self.data.index, drawdown, 0, color='red', alpha=0.3)
        ax3.plot(self.data.index, drawdown, color='red', linewidth=1)
        ax3.set_title('Strategy Drawdown', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Drawdown (%)', fontsize=12)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('moving_average_crossover_strategy.png')
        
    def run_full_backtest(self):
        """Execute the complete backtesting workflow"""
        self.fetch_data()
        self.generate_signals()
        self.backtest_strategy()
        self.calculate_performance_metrics()
        self.plot_results()
        
        return self.data, self.trade_log


# =============================================================================
# EXECUTE THE STRATEGY
# =============================================================================

if __name__ == "__main__":
    print("🚀 Bitcoin Moving Average Crossover Strategy Backtest")
    print("=" * 60)
    
    # Initialize and run the strategy
    strategy = MovingAverageCrossoverStrategy(
        ticker='ETH-USD',
        initial_capital=10000,
        short_window=20,
        long_window=100
    )
    
    # Run the complete backtest
    data, trades = strategy.run_full_backtest()
    
    print("\n🎉 Backtest Analysis Complete!")
    print("\nTrade Log (First 10 trades):")
    print(trades.head(10).to_string(index=False))