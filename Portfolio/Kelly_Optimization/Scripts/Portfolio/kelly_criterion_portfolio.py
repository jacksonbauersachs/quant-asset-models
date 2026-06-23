import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class KellyPortfolioOptimizer:
    def __init__(self):
        # Asset parameters from our models
        self.assets = {
            'Bitcoin': {
                'expected_return': 0.2131,  # 21.31% 10-year CAGR from our model
                'volatility': 0.60,         # High volatility (estimated)
                'sharpe_ratio': None        # Will calculate
            },
            'SP500': {
                'expected_return': 0.0778,  # 7.78% 10-year CAGR from our model
                'volatility': 0.16,         # ~16% historical volatility
                'sharpe_ratio': None        # Will calculate
            }
        }
        
        # Risk-free rate (more realistic 3% for long-term planning)
        self.risk_free_rate = 0.03
        
        # Calculate Sharpe ratios
        for asset in self.assets:
            excess_return = self.assets[asset]['expected_return'] - self.risk_free_rate
            self.assets[asset]['sharpe_ratio'] = excess_return / self.assets[asset]['volatility']
    
    def kelly_single_asset(self, asset_name):
        """Calculate Kelly fraction for a single asset (capped at 100%)"""
        asset = self.assets[asset_name]
        excess_return = asset['expected_return'] - self.risk_free_rate
        variance = asset['volatility'] ** 2
        
        # Kelly fraction = (μ - r) / σ²
        kelly_fraction = excess_return / variance
        
        # Cap at 100% (no leverage for individual investors)
        kelly_fraction = min(kelly_fraction, 1.0)
        
        return kelly_fraction
    
    def kelly_two_asset_portfolio(self, correlation=0.3):
        """Calculate optimal Kelly allocation between Bitcoin and S&P 500"""
        
        # Asset parameters
        mu1 = self.assets['Bitcoin']['expected_return'] - self.risk_free_rate
        mu2 = self.assets['SP500']['expected_return'] - self.risk_free_rate
        sigma1 = self.assets['Bitcoin']['volatility']
        sigma2 = self.assets['SP500']['volatility']
        
        # Covariance matrix
        var1 = sigma1 ** 2
        var2 = sigma2 ** 2
        cov12 = correlation * sigma1 * sigma2
        
        # Kelly optimal weights (analytical solution)
        denominator = var1 * var2 - cov12 ** 2
        
        w1 = (mu1 * var2 - mu2 * cov12) / denominator  # Bitcoin weight
        w2 = (mu2 * var1 - mu1 * cov12) / denominator  # S&P 500 weight
        
        # Cap total allocation at 100% (no leverage)
        total_weight = w1 + w2
        if total_weight > 1.0:
            w1 = w1 / total_weight
            w2 = w2 / total_weight
        
        return w1, w2
    
    def fractional_kelly(self, full_kelly_weights, fraction=0.25):
        """Apply fractional Kelly to reduce risk"""
        return [w * fraction for w in full_kelly_weights]
    
    def portfolio_metrics(self, w1, w2, correlation=0.3):
        """Calculate portfolio expected return and volatility"""
        mu1 = self.assets['Bitcoin']['expected_return']
        mu2 = self.assets['SP500']['expected_return']
        sigma1 = self.assets['Bitcoin']['volatility']
        sigma2 = self.assets['SP500']['volatility']
        
        # Portfolio expected return
        portfolio_return = w1 * mu1 + w2 * mu2
        
        # Portfolio volatility
        portfolio_variance = (w1**2 * sigma1**2 + 
                            w2**2 * sigma2**2 + 
                            2 * w1 * w2 * correlation * sigma1 * sigma2)
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Portfolio Sharpe ratio
        portfolio_sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        return portfolio_return, portfolio_volatility, portfolio_sharpe
    
    def run_analysis(self):
        """Run complete Kelly Criterion analysis"""
        print("KELLY CRITERION PORTFOLIO OPTIMIZATION")
        print("=" * 60)
        
        # Single asset Kelly fractions
        print("\n1. SINGLE ASSET KELLY FRACTIONS:")
        print("-" * 40)
        for asset_name in self.assets:
            kelly_frac = self.kelly_single_asset(asset_name)
            asset = self.assets[asset_name]
            print(f"{asset_name}:")
            print(f"  Expected Return: {asset['expected_return']:.2%}")
            print(f"  Volatility: {asset['volatility']:.2%}")
            print(f"  Sharpe Ratio: {asset['sharpe_ratio']:.3f}")
            print(f"  Kelly Fraction: {kelly_frac:.1%}")
            print()
        
        # Two-asset portfolio optimization
        print("2. TWO-ASSET PORTFOLIO OPTIMIZATION:")
        print("-" * 40)
        
        correlations = [0.1, 0.3, 0.5]  # Different correlation scenarios
        
        for corr in correlations:
            print(f"\nCorrelation = {corr:.1f}:")
            w1, w2 = self.kelly_two_asset_portfolio(corr)
            
            # Full Kelly
            total_leverage = w1 + w2
            portfolio_return, portfolio_vol, portfolio_sharpe = self.portfolio_metrics(w1, w2, corr)
            
            print(f"  Full Kelly Allocation:")
            print(f"    Bitcoin: {w1:.1%}")
            print(f"    S&P 500: {w2:.1%}")
            print(f"    Total Leverage: {total_leverage:.1%}")
            print(f"    Expected Return: {portfolio_return:.2%}")
            print(f"    Portfolio Volatility: {portfolio_vol:.2%}")
            print(f"    Sharpe Ratio: {portfolio_sharpe:.3f}")
            
            # Fractional Kelly (25% of full Kelly)
            w1_frac, w2_frac = self.fractional_kelly([w1, w2], 0.25)
            cash_weight = 1 - w1_frac - w2_frac
            
            portfolio_return_frac, portfolio_vol_frac, portfolio_sharpe_frac = self.portfolio_metrics(w1_frac, w2_frac, corr)
            
            print(f"  Fractional Kelly (25%):")
            print(f"    Bitcoin: {w1_frac:.1%}")
            print(f"    S&P 500: {w2_frac:.1%}")
            print(f"    Cash: {cash_weight:.1%}")
            print(f"    Expected Return: {portfolio_return_frac:.2%}")
            print(f"    Portfolio Volatility: {portfolio_vol_frac:.2%}")
            print(f"    Sharpe Ratio: {portfolio_sharpe_frac:.3f}")
        
        # Practical recommendations
        print("\n3. PRACTICAL RECOMMENDATIONS:")
        print("-" * 40)
        
        # Use moderate correlation (0.3) for recommendation
        w1_opt, w2_opt = self.kelly_two_asset_portfolio(0.3)
        
        # Different risk tolerance levels
        risk_levels = {
            'Conservative': 0.1,
            'Moderate': 0.25,
            'Aggressive': 0.5
        }
        
        for risk_name, fraction in risk_levels.items():
            w1_risk, w2_risk = self.fractional_kelly([w1_opt, w2_opt], fraction)
            cash_weight = 1 - w1_risk - w2_risk
            
            portfolio_return, portfolio_vol, portfolio_sharpe = self.portfolio_metrics(w1_risk, w2_risk, 0.3)
            
            print(f"\n{risk_name} Portfolio ({fraction:.0%} Kelly):")
            print(f"  Bitcoin: {w1_risk:.1%}")
            print(f"  S&P 500: {w2_risk:.1%}")
            print(f"  Cash: {cash_weight:.1%}")
            print(f"  Expected Return: {portfolio_return:.2%}")
            print(f"  Volatility: {portfolio_vol:.2%}")
            print(f"  Sharpe Ratio: {portfolio_sharpe:.3f}")
        
        # Investment decision framework
        print("\n4. INVESTMENT DECISION FRAMEWORK:")
        print("-" * 40)
        print("Based on Kelly Criterion analysis:")
        print(f"• Bitcoin has higher expected return ({self.assets['Bitcoin']['expected_return']:.2%})")
        print(f"• But much higher volatility ({self.assets['Bitcoin']['volatility']:.2%})")
        print(f"• S&P 500 has lower return ({self.assets['SP500']['expected_return']:.2%})")
        print(f"• But much lower volatility ({self.assets['SP500']['volatility']:.2%})")
        print()
        print("Optimal strategy depends on your risk tolerance:")
        print("• Conservative: Heavy S&P 500, small Bitcoin allocation")
        print("• Moderate: Balanced approach with both assets")
        print("• Aggressive: Higher Bitcoin allocation (but still diversified)")
        print()
        print("⚠️  WARNING: Full Kelly can be very aggressive!")
        print("   Most professionals use 10-25% of full Kelly allocation.")

def main():
    optimizer = KellyPortfolioOptimizer()
    optimizer.run_analysis()

if __name__ == "__main__":
    main() 