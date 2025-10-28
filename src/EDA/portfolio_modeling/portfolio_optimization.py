"""
Portfolio Optimization Analysis Script
=====================================
This script performs portfolio optimization using various strategies:
- Global Minimum Variance (GMV)
- Global Maximum Sharpe (GMS) 
- Global Maximum Return (GMR)
- Equal Weight (EW)

Dependencies: numpy, pandas, riskfolio, yfinance
"""

import numpy as np
import pandas as pd
import riskfolio as rp
import yfinance as yf

def print_header(title, char="="):
    """Print a formatted header"""
    print(f"\n{char * 60}")
    print(f"{title:^60}")
    print(f"{char * 60}")

def print_section(title, char="-"):
    """Print a formatted section header"""
    print(f"\n{char * 40}")
    print(f"{title}")
    print(f"{char * 40}")

def main():
    print_header("PORTFOLIO OPTIMIZATION ANALYSIS")
    
    # Define portfolio parameters
    print_section("Portfolio Configuration")
    tickers = ['CSSPX.MI', 'IEUR', 'WSML.L', 'VFEA.MI', 'EWJ', 'VGEK.DE']
    start_date = '2000-01-01'
    
    print(f"Selected Assets: {', '.join(tickers)}")
    print(f"Start Date: {start_date}")
    print(f"Number of Assets: {len(tickers)}")
    
    # Download and process data
    print_section("Data Download and Processing")
    
    try:
        prices = yf.download(tickers, start=start_date, auto_adjust=True)['Close']
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        monthly_returns = returns.resample('ME').agg(lambda x: (1 + x).prod() - 1)
        
        print(f"Data range: {monthly_returns.index[0].strftime('%Y-%m-%d')} to {monthly_returns.index[-1].strftime('%Y-%m-%d')}")
        print(f"Total monthly observations: {len(monthly_returns)}")
        
    except Exception as e:
        print(f"✗ Error downloading data: {e}")
        return
    
    # Calculate individual asset statistics
    print_section("Individual Asset Analysis")
    asset_vol = monthly_returns.std() * np.sqrt(12)
    asset_vol = asset_vol.rename('Ann.Vol')
    asset_returns = (1 + monthly_returns.mean())**12 - 1  
    asset_sharpe = asset_returns / asset_vol
    
    asset_results = pd.DataFrame({
        'Annualized_volatility': asset_vol,
        'Annualized_return': asset_returns,
        'Sharpe_ratio': asset_sharpe
    })
    
    print(asset_results.round(4))
    
    
    portfolio = rp.Portfolio(returns=monthly_returns)
    portfolio.assets_stats(method_mu='hist', method_cov='hist')
    
    # Initialize results storage
    optimization_results = {}
    
    # Global Minimum Variance Portfolio
    try:
        gmv = portfolio.optimization(model='Classic', rm='MV', obj='MinRisk', rf=0)
        if gmv is not None:
            gmv.name = "Global Minimum Variance Portfolio"
            gmv_vol = np.sqrt(gmv.weights @ monthly_returns.cov() @ gmv.weights.T) * np.sqrt(12)
            gmv_ret = (1 + (monthly_returns @ gmv.weights).mean())**12 - 1  
            gmv_sharpe = gmv_ret / gmv_vol
            optimization_results['GMV'] = {
                'weights': gmv.weights,
                'vol': gmv_vol,
                'ret': gmv_ret,
                'sharpe': gmv_sharpe
            }
        else:
            print("✗ GMV optimization failed")
            optimization_results['GMV'] = None
    except Exception as e:
        print(f"✗ GMV optimization error: {e}")
        optimization_results['GMV'] = None
    
    # Global Maximum Sharpe Portfolio
    try:
        gms = portfolio.optimization(model='Classic', rm='MV', obj='Sharpe', rf=0)
        if gms is not None:
            gms.name = "Global Maximum Sharpe Portfolio"
            gms_vol = np.sqrt(gms.weights @ monthly_returns.cov() @ gms.weights.T) * np.sqrt(12)
            gms_ret = (1 + (monthly_returns @ gms.weights).mean())**12 - 1
            gms_sharpe = gms_ret / gms_vol
            optimization_results['GMS'] = {
                'weights': gms.weights,
                'vol': gms_vol,
                'ret': gms_ret,
                'sharpe': gms_sharpe
            }
        else:
            print("✗ GMS optimization failed")
            optimization_results['GMS'] = None
    except Exception as e:
        print(f"✗ GMS optimization error: {e}")
        optimization_results['GMS'] = None
    
    # Global Maximum Return Portfolio
    try:
        gmr = portfolio.optimization(model='Classic', rm='MV', obj='MaxRet', rf=0)
        if gmr is not None:
            gmr.name = "Global Maximum Return Portfolio"
            gmr_vol = np.sqrt(gmr.weights @ monthly_returns.cov() @ gmr.weights.T) * np.sqrt(12)
            gmr_ret = (1 + (monthly_returns @ gmr.weights).mean())**12 - 1
            gmr_sharpe = gmr_ret / gmr_vol
            optimization_results['GMR'] = {
                'weights': gmr.weights,
                'vol': gmr_vol,
                'ret': gmr_ret,
                'sharpe': gmr_sharpe
            }
        else:
            print("✗ GMR optimization failed")
            optimization_results['GMR'] = None
    except Exception as e:
        print(f"✗ GMR optimization error: {e}")
        optimization_results['GMR'] = None
    
    # Equal Weight Portfolio (1/N strategy)
    try:
        ew_weights = pd.DataFrame(np.ones((len(tickers), 1)) / len(tickers), 
                                 index=monthly_returns.columns, columns=['weights'])
        ew_vol = np.sqrt(ew_weights.weights.T @ monthly_returns.cov() @ ew_weights.weights) * np.sqrt(12)
        ew_ret = (1 + (monthly_returns @ ew_weights.weights).mean())**12 - 1
        ew_sharpe = ew_ret / ew_vol
        optimization_results['EW'] = {
            'weights': ew_weights.weights,
            'vol': ew_vol,
            'ret': ew_ret,
            'sharpe': ew_sharpe
        }
    except Exception as e:
        print(f"✗ Equal Weight portfolio error: {e}")
        optimization_results['EW'] = None
    
    # Create comprehensive results table
    print_section("Portfolio Optimization Results")
    
    portfolio_results = pd.DataFrame({
        'Annualized_volatility_%': pd.Series({
            'Portfolio(GMV)': round(optimization_results['GMV']['vol'] * 100, 2) if optimization_results['GMV'] else None,
            'Portfolio(GMS)': round(optimization_results['GMS']['vol'] * 100, 2) if optimization_results['GMS'] else None,
            'Portfolio(GMR)': round(optimization_results['GMR']['vol'] * 100, 2) if optimization_results['GMR'] else None,
            'Portfolio(EW)': round(optimization_results['EW']['vol'] * 100, 2) if optimization_results['EW'] else None
        }),
        'Annualized_return_%': pd.Series({
            'Portfolio(GMV)': round(optimization_results['GMV']['ret'] * 100, 2) if optimization_results['GMV'] else None,
            'Portfolio(GMS)': round(optimization_results['GMS']['ret'] * 100, 2) if optimization_results['GMS'] else None,
            'Portfolio(GMR)': round(optimization_results['GMR']['ret'] * 100, 2) if optimization_results['GMR'] else None,
            'Portfolio(EW)': round(optimization_results['EW']['ret'] * 100, 2) if optimization_results['EW'] else None
        }),
        'Sharpe_ratio': pd.Series({
            'Portfolio(GMV)': round(optimization_results['GMV']['sharpe'], 3) if optimization_results['GMV'] else None,
            'Portfolio(GMS)': round(optimization_results['GMS']['sharpe'], 3) if optimization_results['GMS'] else None,
            'Portfolio(GMR)': round(optimization_results['GMR']['sharpe'], 3) if optimization_results['GMR'] else None,
            'Portfolio(EW)': round(optimization_results['EW']['sharpe'], 3) if optimization_results['EW'] else None
        })
    })
    
    # Add weights as percentages
    for ticker in monthly_returns.columns:
        portfolio_results[f'{ticker}_weight_%'] = pd.Series({
            'Portfolio(GMV)': round(optimization_results['GMV']['weights'][ticker] * 100, 2) if optimization_results['GMV'] else None,
            'Portfolio(GMS)': round(optimization_results['GMS']['weights'][ticker] * 100, 2) if optimization_results['GMS'] else None,
            'Portfolio(GMR)': round(optimization_results['GMR']['weights'][ticker] * 100, 2) if optimization_results['GMR'] else None,
            'Portfolio(EW)': round(optimization_results['EW']['weights'][ticker] * 100, 2) if optimization_results['EW'] else None
        })
    
    print("Portfolio Performance Summary:")
    print(portfolio_results[['Annualized_volatility_%', 'Annualized_return_%', 'Sharpe_ratio']])

    # Print portfolio weights separately for better readability
    print_section("Portfolio Weight Allocations (%)")
    weight_columns = [col for col in portfolio_results.columns if '_weight_%' in col]
    weights_df = portfolio_results[weight_columns].copy()
    weights_df.columns = [col.replace('_weight_%', '') for col in weights_df.columns]
    print(weights_df)

if __name__ == "__main__":
    main()
