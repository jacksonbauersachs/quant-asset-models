import numpy as np
from scipy.optimize import minimize

class NoLeverageKelly:
    def __init__(self):
        # Asset parameters from our models
        self.bitcoin_return = 0.2131  # 21.31%
        self.bitcoin_vol = 0.60       # 60%
        self.sp500_return = 0.0778    # 7.78%
        self.sp500_vol = 0.16         # 16%
        self.risk_free_rate = 0.03    # 3%
        
        # Calculate excess returns
        self.bitcoin_excess = self.bitcoin_return - self.risk_free_rate
        self.sp500_excess = self.sp500_return - self.risk_free_rate
        
        print("NO-LEVERAGE KELLY CRITERION CALCULATOR")
        print("=" * 50)
        print(f"Bitcoin: {self.bitcoin_return:.2%} return, {self.bitcoin_vol:.0%} volatility")
        print(f"S&P 500: {self.sp500_return:.2%} return, {self.sp500_vol:.0%} volatility")
        print(f"Risk-free rate: {self.risk_free_rate:.1%}")
        print(f"Bitcoin excess return: {self.bitcoin_excess:.2%}")
        print(f"S&P 500 excess return: {self.sp500_excess:.2%}")
        print()
    
    def kelly_objective(self, weights, correlation):
        """
        Kelly objective function to maximize: w^T * μ - 0.5 * w^T * Σ * w
        where w = weights, μ = excess returns, Σ = covariance matrix
        """
        w_btc, w_sp500 = weights
        w_cash = 1 - w_btc - w_sp500
        
        # Expected excess return of portfolio
        portfolio_excess_return = w_btc * self.bitcoin_excess + w_sp500 * self.sp500_excess
        
        # Portfolio variance
        covariance = correlation * self.bitcoin_vol * self.sp500_vol
        portfolio_variance = (w_btc**2 * self.bitcoin_vol**2 + 
                            w_sp500**2 * self.sp500_vol**2 + 
                            2 * w_btc * w_sp500 * covariance)
        
        # Kelly objective (we minimize the negative to maximize)
        kelly_objective = portfolio_excess_return - 0.5 * portfolio_variance
        
        return -kelly_objective  # Negative because we're minimizing
    
    def optimize_no_leverage(self, correlation=0.3):
        """
        Optimize Kelly allocation with no leverage constraint
        """
        # Constraints: weights sum to ≤ 1, all weights ≥ 0
        constraints = [
            {'type': 'ineq', 'fun': lambda w: 1 - w[0] - w[1]},  # w_btc + w_sp500 ≤ 1
            {'type': 'ineq', 'fun': lambda w: w[0]},              # w_btc ≥ 0
            {'type': 'ineq', 'fun': lambda w: w[1]}               # w_sp500 ≥ 0
        ]
        
        # Initial guess
        x0 = [0.3, 0.5]  # Start with 30% Bitcoin, 50% S&P 500
        
        # Optimize
        result = minimize(
            self.kelly_objective,
            x0,
            args=(correlation,),
            method='SLSQP',
            constraints=constraints,
            bounds=[(0, 1), (0, 1)]  # Each weight between 0 and 1
        )
        
        return result.x
    
    def calculate_portfolio_metrics(self, w_btc, w_sp500, correlation):
        """Calculate portfolio return, volatility, and Sharpe ratio"""
        w_cash = 1 - w_btc - w_sp500
        
        # Portfolio return
        portfolio_return = (w_btc * self.bitcoin_return + 
                          w_sp500 * self.sp500_return + 
                          w_cash * self.risk_free_rate)
        
        # Portfolio volatility
        covariance = correlation * self.bitcoin_vol * self.sp500_vol
        portfolio_variance = (w_btc**2 * self.bitcoin_vol**2 + 
                            w_sp500**2 * self.sp500_vol**2 + 
                            2 * w_btc * w_sp500 * covariance)
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Sharpe ratio
        portfolio_excess = portfolio_return - self.risk_free_rate
        sharpe_ratio = portfolio_excess / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def run_analysis(self):
        """Run complete no-leverage Kelly analysis"""
        print("NO-LEVERAGE KELLY OPTIMIZATION RESULTS")
        print("=" * 50)
        
        correlations = [0.1, 0.3, 0.5]
        
        for corr in correlations:
            print(f"\nCorrelation = {corr:.1f}:")
            print("-" * 30)
            
            # Optimize
            optimal_weights = self.optimize_no_leverage(corr)
            w_btc, w_sp500 = optimal_weights
            w_cash = 1 - w_btc - w_sp500
            
            # Calculate metrics
            portfolio_return, portfolio_vol, sharpe = self.calculate_portfolio_metrics(
                w_btc, w_sp500, corr
            )
            
            print(f"Optimal Allocation:")
            print(f"  Bitcoin: {w_btc:.1%}")
            print(f"  S&P 500: {w_sp500:.1%}")
            print(f"  Cash: {w_cash:.1%}")
            print(f"  Total: {w_btc + w_sp500 + w_cash:.1%}")
            print()
            print(f"Portfolio Metrics:")
            print(f"  Expected Return: {portfolio_return:.2%}")
            print(f"  Volatility: {portfolio_vol:.2%}")
            print(f"  Sharpe Ratio: {sharpe:.3f}")
            print()
        
        # Compare with unconstrained Kelly
        print("COMPARISON WITH UNCONSTRAINED KELLY:")
        print("=" * 50)
        
        corr = 0.3  # Use moderate correlation
        
        # Unconstrained Kelly (can go over 100%)
        bitcoin_kelly_single = self.bitcoin_excess / (self.bitcoin_vol**2)
        sp500_kelly_single = self.sp500_excess / (self.sp500_vol**2)
        
        print(f"Single Asset Kelly Fractions:")
        print(f"  Bitcoin: {bitcoin_kelly_single:.1%}")
        print(f"  S&P 500: {sp500_kelly_single:.1%}")
        print()
        
        # Constrained Kelly
        optimal_weights = self.optimize_no_leverage(corr)
        w_btc, w_sp500 = optimal_weights
        w_cash = 1 - w_btc - w_sp500
        
        print(f"No-Leverage Kelly Allocation:")
        print(f"  Bitcoin: {w_btc:.1%}")
        print(f"  S&P 500: {w_sp500:.1%}")
        print(f"  Cash: {w_cash:.1%}")
        print()
        
        # Show the difference
        print("KEY INSIGHTS:")
        print("-" * 30)
        print("• No-leverage constraint forces more conservative allocation")
        print("• Cash allocation acts as a 'safety buffer'")
        print("• Still maintains optimal risk-adjusted returns without debt")
        print("• More practical for individual investors")
        
        return optimal_weights

def main():
    calculator = NoLeverageKelly()
    optimal_weights = calculator.run_analysis()
    
    print(f"\n🎯 FINAL RECOMMENDATION (No Leverage):")
    print(f"   Bitcoin: {optimal_weights[0]:.1%}")
    print(f"   S&P 500: {optimal_weights[1]:.1%}")
    print(f"   Cash: {1 - optimal_weights[0] - optimal_weights[1]:.1%}")
    
    # Save results to file
    with open('Investment Strategy Analasis/no_leverage_kelly_results.txt', 'w') as f:
        f.write("NO-LEVERAGE KELLY CRITERION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("INPUT PARAMETERS:\n")
        f.write(f"Bitcoin Expected Return: {calculator.bitcoin_return:.2%}\n")
        f.write(f"Bitcoin Volatility: {calculator.bitcoin_vol:.0%}\n")
        f.write(f"S&P 500 Expected Return: {calculator.sp500_return:.2%}\n")
        f.write(f"S&P 500 Volatility: {calculator.sp500_vol:.0%}\n")
        f.write(f"Risk-Free Rate: {calculator.risk_free_rate:.1%}\n\n")
        
        f.write("OPTIMAL ALLOCATION (No Leverage):\n")
        f.write(f"Bitcoin: {optimal_weights[0]:.1%}\n")
        f.write(f"S&P 500: {optimal_weights[1]:.1%}\n")
        f.write(f"Cash: {1 - optimal_weights[0] - optimal_weights[1]:.1%}\n\n")
        
        # Calculate portfolio metrics
        portfolio_return, portfolio_vol, sharpe = calculator.calculate_portfolio_metrics(
            optimal_weights[0], optimal_weights[1], 0.3
        )
        
        f.write("PORTFOLIO PERFORMANCE:\n")
        f.write(f"Expected Return: {portfolio_return:.2%}\n")
        f.write(f"Volatility: {portfolio_vol:.2%}\n")
        f.write(f"Sharpe Ratio: {sharpe:.3f}\n\n")
        
        f.write("INVESTMENT STRATEGY:\n")
        f.write("For every $100 invested:\n")
        f.write(f"- Put ${optimal_weights[0]*100:.0f} in Bitcoin\n")
        f.write(f"- Put ${optimal_weights[1]*100:.0f} in S&P 500\n")
        f.write(f"- Keep ${(1-optimal_weights[0]-optimal_weights[1])*100:.0f} in cash\n\n")
        
        f.write("This allocation maximizes risk-adjusted returns without using leverage.\n")
    
    print(f"\n📁 Results saved to: Investment Strategy Analasis/no_leverage_kelly_results.txt")

if __name__ == "__main__":
    main() 