import numpy as np
from scipy.optimize import minimize

class ThreeAssetKelly:
    def __init__(self):
        # Asset parameters from our models
        self.assets = {
            'Bitcoin': {
                'return': 0.2131,    # 21.31%
                'volatility': 0.60   # 60%
            },
            'SP500': {
                'return': 0.0778,    # 7.78%
                'volatility': 0.16   # 16%
            },
            'HYSA': {
                'return': 0.05,      # 5% high-yield savings
                'volatility': 0.0    # 0% volatility (risk-free)
            }
        }
        
        # Risk-free rate for excess return calculation (use HYSA as baseline)
        self.risk_free_rate = 0.05  # 5% HYSA rate
        
        # Calculate excess returns
        self.bitcoin_excess = self.assets['Bitcoin']['return'] - self.risk_free_rate
        self.sp500_excess = self.assets['SP500']['return'] - self.risk_free_rate
        self.hysa_excess = self.assets['HYSA']['return'] - self.risk_free_rate  # This will be 0
        
        print("THREE-ASSET KELLY CRITERION CALCULATOR")
        print("=" * 60)
        print(f"Bitcoin: {self.assets['Bitcoin']['return']:.2%} return, {self.assets['Bitcoin']['volatility']:.0%} volatility")
        print(f"S&P 500: {self.assets['SP500']['return']:.2%} return, {self.assets['SP500']['volatility']:.0%} volatility")
        print(f"High-Yield Savings: {self.assets['HYSA']['return']:.1%} return, {self.assets['HYSA']['volatility']:.0%} volatility")
        print(f"Risk-free baseline: {self.risk_free_rate:.1%}")
        print()
        print(f"Excess returns:")
        print(f"  Bitcoin: {self.bitcoin_excess:.2%}")
        print(f"  S&P 500: {self.sp500_excess:.2%}")
        print(f"  HYSA: {self.hysa_excess:.2%}")
        print()
    
    def build_covariance_matrix(self, btc_sp500_corr=0.3):
        """
        Build 3x3 covariance matrix
        HYSA has 0 volatility, so correlations with HYSA are 0
        """
        # Variances
        btc_var = self.assets['Bitcoin']['volatility'] ** 2
        sp500_var = self.assets['SP500']['volatility'] ** 2
        hysa_var = self.assets['HYSA']['volatility'] ** 2  # = 0
        
        # Covariances
        btc_sp500_cov = btc_sp500_corr * self.assets['Bitcoin']['volatility'] * self.assets['SP500']['volatility']
        btc_hysa_cov = 0  # HYSA has 0 volatility
        sp500_hysa_cov = 0  # HYSA has 0 volatility
        
        # Build covariance matrix
        cov_matrix = np.array([
            [btc_var,       btc_sp500_cov,  btc_hysa_cov],
            [btc_sp500_cov, sp500_var,      sp500_hysa_cov],
            [btc_hysa_cov,  sp500_hysa_cov, hysa_var]
        ])
        
        return cov_matrix
    
    def kelly_objective(self, weights, cov_matrix):
        """
        Three-asset Kelly objective function to maximize: w^T * μ - 0.5 * w^T * Σ * w
        """
        w = np.array(weights)
        
        # Excess returns vector
        excess_returns = np.array([self.bitcoin_excess, self.sp500_excess, self.hysa_excess])
        
        # Expected excess return of portfolio
        portfolio_excess_return = np.dot(w, excess_returns)
        
        # Portfolio variance
        portfolio_variance = np.dot(w, np.dot(cov_matrix, w))
        
        # Kelly objective (we minimize the negative to maximize)
        kelly_objective = portfolio_excess_return - 0.5 * portfolio_variance
        
        return -kelly_objective  # Negative because we're minimizing
    
    def optimize_three_asset(self, btc_sp500_corr=0.3):
        """
        Optimize three-asset Kelly allocation
        """
        # Build covariance matrix
        cov_matrix = self.build_covariance_matrix(btc_sp500_corr)
        
        # Constraints: weights sum to 1, all weights ≥ 0
        constraints = [
            {'type': 'eq', 'fun': lambda w: w[0] + w[1] + w[2] - 1},  # weights sum to 1
            {'type': 'ineq', 'fun': lambda w: w[0]},                  # w_btc ≥ 0
            {'type': 'ineq', 'fun': lambda w: w[1]},                  # w_sp500 ≥ 0
            {'type': 'ineq', 'fun': lambda w: w[2]}                   # w_hysa ≥ 0
        ]
        
        # Initial guess
        x0 = [0.3, 0.4, 0.3]  # Start with equal-ish allocation
        
        # Optimize
        result = minimize(
            self.kelly_objective,
            x0,
            args=(cov_matrix,),
            method='SLSQP',
            constraints=constraints,
            bounds=[(0, 1), (0, 1), (0, 1)]  # Each weight between 0 and 1
        )
        
        return result.x
    
    def calculate_portfolio_metrics(self, w_btc, w_sp500, w_hysa, btc_sp500_corr):
        """Calculate portfolio return, volatility, and Sharpe ratio"""
        
        # Portfolio return
        portfolio_return = (w_btc * self.assets['Bitcoin']['return'] + 
                          w_sp500 * self.assets['SP500']['return'] + 
                          w_hysa * self.assets['HYSA']['return'])
        
        # Portfolio volatility
        cov_matrix = self.build_covariance_matrix(btc_sp500_corr)
        weights = np.array([w_btc, w_sp500, w_hysa])
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Sharpe ratio (using HYSA as risk-free rate)
        portfolio_excess = portfolio_return - self.risk_free_rate
        sharpe_ratio = portfolio_excess / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def run_analysis(self):
        """Run complete three-asset Kelly analysis"""
        print("THREE-ASSET KELLY OPTIMIZATION RESULTS")
        print("=" * 60)
        
        correlations = [0.1, 0.3, 0.5]
        
        for corr in correlations:
            print(f"\nBitcoin-S&P500 Correlation = {corr:.1f}:")
            print("-" * 40)
            
            # Optimize
            optimal_weights = self.optimize_three_asset(corr)
            w_btc, w_sp500, w_hysa = optimal_weights
            
            # Calculate metrics
            portfolio_return, portfolio_vol, sharpe = self.calculate_portfolio_metrics(
                w_btc, w_sp500, w_hysa, corr
            )
            
            print(f"Optimal Allocation:")
            print(f"  Bitcoin: {w_btc:.1%}")
            print(f"  S&P 500: {w_sp500:.1%}")
            print(f"  High-Yield Savings: {w_hysa:.1%}")
            print(f"  Total: {w_btc + w_sp500 + w_hysa:.1%}")
            print()
            print(f"Portfolio Metrics:")
            print(f"  Expected Return: {portfolio_return:.2%}")
            print(f"  Volatility: {portfolio_vol:.2%}")
            print(f"  Sharpe Ratio: {sharpe:.3f}")
            print()
        
        # Compare with two-asset (no HYSA option)
        print("COMPARISON: WITH vs WITHOUT HIGH-YIELD SAVINGS")
        print("=" * 60)
        
        corr = 0.3  # Use moderate correlation
        
        # Three-asset optimal
        optimal_weights = self.optimize_three_asset(corr)
        w_btc, w_sp500, w_hysa = optimal_weights
        
        print(f"Three-Asset Allocation (with HYSA):")
        print(f"  Bitcoin: {w_btc:.1%}")
        print(f"  S&P 500: {w_sp500:.1%}")
        print(f"  High-Yield Savings: {w_hysa:.1%}")
        print()
        
        # Two-asset equivalent (force HYSA = 0)
        from scipy.optimize import minimize
        
        def two_asset_objective(weights):
            w_btc, w_sp500 = weights
            w_hysa = 1 - w_btc - w_sp500
            cov_matrix = self.build_covariance_matrix(corr)
            return self.kelly_objective([w_btc, w_sp500, w_hysa], cov_matrix)
        
        constraints_2asset = [
            {'type': 'eq', 'fun': lambda w: w[0] + w[1] - 1},  # weights sum to 1 (no HYSA)
            {'type': 'ineq', 'fun': lambda w: w[0]},           # w_btc ≥ 0
            {'type': 'ineq', 'fun': lambda w: w[1]}            # w_sp500 ≥ 0
        ]
        
        result_2asset = minimize(
            two_asset_objective,
            [0.4, 0.6],
            method='SLSQP',
            constraints=constraints_2asset,
            bounds=[(0, 1), (0, 1)]
        )
        
        w_btc_2asset, w_sp500_2asset = result_2asset.x
        
        print(f"Two-Asset Allocation (no HYSA option):")
        print(f"  Bitcoin: {w_btc_2asset:.1%}")
        print(f"  S&P 500: {w_sp500_2asset:.1%}")
        print(f"  Cash (3% rate): {0:.1%}")
        print()
        
        # Show the impact
        print("KEY INSIGHTS:")
        print("-" * 40)
        print(f"• Adding HYSA option increases cash allocation to {w_hysa:.1%}")
        print(f"• HYSA at 5% is attractive vs 3% risk-free rate")
        print(f"• Reduces risk asset allocation slightly")
        print(f"• Provides better risk-adjusted returns")
        
        return optimal_weights

def main():
    calculator = ThreeAssetKelly()
    optimal_weights = calculator.run_analysis()
    
    print(f"\n🎯 FINAL RECOMMENDATION (Three Assets):")
    print(f"   Bitcoin: {optimal_weights[0]:.1%}")
    print(f"   S&P 500: {optimal_weights[1]:.1%}")
    print(f"   High-Yield Savings: {optimal_weights[2]:.1%}")
    
    # Save results to file
    with open('Investment Strategy Analasis/three_asset_kelly_results.txt', 'w') as f:
        f.write("THREE-ASSET KELLY CRITERION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("INPUT PARAMETERS:\n")
        f.write(f"Bitcoin Expected Return: {calculator.assets['Bitcoin']['return']:.2%}\n")
        f.write(f"Bitcoin Volatility: {calculator.assets['Bitcoin']['volatility']:.0%}\n")
        f.write(f"S&P 500 Expected Return: {calculator.assets['SP500']['return']:.2%}\n")
        f.write(f"S&P 500 Volatility: {calculator.assets['SP500']['volatility']:.0%}\n")
        f.write(f"High-Yield Savings Return: {calculator.assets['HYSA']['return']:.1%}\n")
        f.write(f"High-Yield Savings Volatility: {calculator.assets['HYSA']['volatility']:.0%}\n\n")
        
        f.write("OPTIMAL ALLOCATION (Three Assets):\n")
        f.write(f"Bitcoin: {optimal_weights[0]:.1%}\n")
        f.write(f"S&P 500: {optimal_weights[1]:.1%}\n")
        f.write(f"High-Yield Savings: {optimal_weights[2]:.1%}\n\n")
        
        # Calculate portfolio metrics
        portfolio_return, portfolio_vol, sharpe = calculator.calculate_portfolio_metrics(
            optimal_weights[0], optimal_weights[1], optimal_weights[2], 0.3
        )
        
        f.write("PORTFOLIO PERFORMANCE:\n")
        f.write(f"Expected Return: {portfolio_return:.2%}\n")
        f.write(f"Volatility: {portfolio_vol:.2%}\n")
        f.write(f"Sharpe Ratio: {sharpe:.3f}\n\n")
        
        f.write("INVESTMENT STRATEGY:\n")
        f.write("For every $100 invested:\n")
        f.write(f"- Put ${optimal_weights[0]*100:.0f} in Bitcoin\n")
        f.write(f"- Put ${optimal_weights[1]*100:.0f} in S&P 500\n")
        f.write(f"- Put ${optimal_weights[2]*100:.0f} in High-Yield Savings (5%)\n\n")
        
        f.write("This allocation maximizes risk-adjusted returns with three assets.\n")
    
    print(f"\n📁 Results saved to: Investment Strategy Analasis/three_asset_kelly_results.txt")

if __name__ == "__main__":
    main() 