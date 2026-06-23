import numpy as np
from scipy.optimize import minimize

class FinalCorrectedKelly:
    def __init__(self):
        """
        FINAL CORRECTED Kelly calculator
        
        The key insight: Your Bitcoin strategy BEATS baseline DCA by 25.4%
        So we need to calculate the effective return of your strategy vs raw Bitcoin
        """
        
        # Baseline DCA performance (calculated from your results)
        # If your strategy returns $69,532 with +25.4% improvement
        # Then baseline DCA = $69,532 / 1.254 = $55,450
        baseline_dca_return = 69532 / 1.254  # ~$55,450
        
        # Your improved strategy performance
        your_strategy_return = 69532  # $69,532 median return
        
        # Calculate the CAGR for each (5-year investment period)
        # Starting with $100k + $1k/month * 60 months = $160k total invested
        total_invested = 100000 + (1000 * 60)  # $160k
        
        # CAGR calculation: (Final/Initial)^(1/years) - 1
        baseline_cagr = (baseline_dca_return / total_invested) ** (1/5) - 1
        your_strategy_cagr = (your_strategy_return / total_invested) ** (1/5) - 1
        
        print(f"📊 PERFORMANCE COMPARISON:")
        print(f"Baseline DCA: ${baseline_dca_return:,.0f} ({baseline_cagr:.2%} CAGR)")
        print(f"Your Strategy: ${your_strategy_return:,.0f} ({your_strategy_cagr:.2%} CAGR)")
        print(f"Improvement: {(your_strategy_return/baseline_dca_return - 1):.1%}")
        print()
        
        # But wait - your strategy has LOWER volatility due to 40% reserves!
        # Raw Bitcoin: 60% volatility
        # Your strategy: 40% reserves (0% vol) + 60% Bitcoin (60% vol)
        # Effective volatility = 0.6 * 60% = 36% volatility
        
        # Strategy definitions for Kelly calculation
        self.btc_strategy = {
            'name': 'Your Bitcoin High-Yield Reserve Strategy',
            'expected_return': your_strategy_cagr,
            'volatility': 0.36,  # Reduced volatility due to 40% reserves
            'description': '40% high-yield savings + 60% tactical Bitcoin'
        }
        
        self.sp500_strategy = {
            'name': 'S&P 500 Strategy',
            'expected_return': 0.0778,  # 7.78% from your model
            'volatility': 0.16,         # 16% volatility
            'description': '100% S&P 500 index fund'
        }
        
        # Raw Bitcoin for comparison
        self.raw_bitcoin = {
            'name': 'Raw Bitcoin',
            'expected_return': 0.2131,  # 21.31% from your growth model
            'volatility': 0.60,         # 60% volatility
            'description': 'Direct Bitcoin investment'
        }
        
        # Correlation
        self.correlation = 0.3
        
        print(f"🎯 KELLY INPUTS:")
        print(f"Your Bitcoin Strategy: {self.btc_strategy['expected_return']:.2%} return, {self.btc_strategy['volatility']:.1%} volatility")
        print(f"S&P 500: {self.sp500_strategy['expected_return']:.2%} return, {self.sp500_strategy['volatility']:.1%} volatility")
        print(f"Raw Bitcoin: {self.raw_bitcoin['expected_return']:.2%} return, {self.raw_bitcoin['volatility']:.1%} volatility")
        print()
    
    def kelly_two_asset_allocation(self, asset1, asset2):
        """
        Calculate Kelly-optimal allocation between two assets
        """
        
        mu1 = asset1['expected_return']
        mu2 = asset2['expected_return']
        sigma1 = asset1['volatility']
        sigma2 = asset2['volatility']
        
        # Covariance matrix
        var1 = sigma1 ** 2
        var2 = sigma2 ** 2
        cov12 = self.correlation * sigma1 * sigma2
        
        def kelly_objective(weights):
            w1, w2 = weights
            
            # Portfolio return
            portfolio_return = w1 * mu1 + w2 * mu2
            
            # Portfolio variance
            portfolio_variance = w1**2 * var1 + w2**2 * var2 + 2 * w1 * w2 * cov12
            
            # Kelly objective (maximize log expected return)
            kelly_value = portfolio_return - 0.5 * portfolio_variance
            
            return -kelly_value
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: w[0] + w[1] - 1},
            {'type': 'ineq', 'fun': lambda w: w[0]},
            {'type': 'ineq', 'fun': lambda w: w[1]}
        ]
        
        result = minimize(
            kelly_objective,
            [0.5, 0.5],
            method='SLSQP',
            constraints=constraints,
            bounds=[(0, 1), (0, 1)]
        )
        
        return result.x[0], result.x[1]
    
    def calculate_portfolio_metrics(self, w1, w2, asset1, asset2):
        """Calculate portfolio metrics"""
        
        portfolio_return = w1 * asset1['expected_return'] + w2 * asset2['expected_return']
        
        covariance = self.correlation * asset1['volatility'] * asset2['volatility']
        portfolio_variance = (w1**2 * asset1['volatility']**2 + 
                            w2**2 * asset2['volatility']**2 + 
                            2 * w1 * w2 * covariance)
        
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        return portfolio_return, portfolio_volatility
    
    def generate_final_analysis(self):
        """Generate the final corrected analysis"""
        
        print("🏆 FINAL CORRECTED KELLY ANALYSIS")
        print("=" * 50)
        print("Comparing your ACTUAL improved Bitcoin strategy vs S&P 500")
        print()
        
        # 1. Your strategy vs S&P 500
        w_your, w_sp500 = self.kelly_two_asset_allocation(self.btc_strategy, self.sp500_strategy)
        your_return, your_vol = self.calculate_portfolio_metrics(w_your, w_sp500, self.btc_strategy, self.sp500_strategy)
        
        print(f"🎯 YOUR STRATEGY vs S&P 500:")
        print(f"  Your Bitcoin Strategy: {w_your:.1%}")
        print(f"  S&P 500: {w_sp500:.1%}")
        print(f"  Expected Return: {your_return:.2%}")
        print(f"  Volatility: {your_vol:.2%}")
        print()
        
        # 2. Raw Bitcoin vs S&P 500 (original Kelly)
        w_raw, w_sp500_raw = self.kelly_two_asset_allocation(self.raw_bitcoin, self.sp500_strategy)
        raw_return, raw_vol = self.calculate_portfolio_metrics(w_raw, w_sp500_raw, self.raw_bitcoin, self.sp500_strategy)
        
        print(f"📊 RAW BITCOIN vs S&P 500 (Original):")
        print(f"  Raw Bitcoin: {w_raw:.1%}")
        print(f"  S&P 500: {w_sp500_raw:.1%}")
        print(f"  Expected Return: {raw_return:.2%}")
        print(f"  Volatility: {raw_vol:.2%}")
        print()
        
        # 3. Comparison
        allocation_change = w_your - w_raw
        
        print(f"🔍 COMPARISON:")
        print(f"  Allocation Change: {allocation_change:+.1%}")
        
        if allocation_change > 0:
            print(f"  ✅ Your improved strategy gets {allocation_change:.1%} MORE allocation!")
            print(f"  🎯 This makes sense - better risk-adjusted returns")
        elif allocation_change < 0:
            print(f"  ❌ Your improved strategy gets {abs(allocation_change):.1%} LESS allocation")
            print(f"  📉 Lower returns don't compensate for the complexity")
        else:
            print(f"  ➖ Same allocation - similar risk-adjusted performance")
        
        print()
        
        # 4. Monthly allocation
        print(f"💰 MONTHLY ALLOCATION ($1,000):")
        print(f"  Your Bitcoin Strategy: ${w_your * 1000:.0f}")
        print(f"  S&P 500: ${w_sp500 * 1000:.0f}")
        print()
        
        if w_your * 1000 > 0:
            reserves = w_your * 1000 * 0.4
            bitcoin_trading = w_your * 1000 * 0.6
            print(f"  Inside Your Bitcoin Strategy:")
            print(f"    → High-Yield Savings: ${reserves:.0f}")
            print(f"    → Bitcoin Trading: ${bitcoin_trading:.0f}")
        
        print()
        print(f"💡 BOTTOM LINE:")
        if allocation_change > 0:
            print(f"  Your improved Bitcoin strategy deserves MORE allocation")
            print(f"  The Kelly criterion confirms your intuition was correct!")
        else:
            print(f"  Your improved Bitcoin strategy needs higher returns to justify more allocation")
            print(f"  Consider optimizing the strategy parameters further")

def main():
    """Run final corrected Kelly analysis"""
    
    calculator = FinalCorrectedKelly()
    calculator.generate_final_analysis()

if __name__ == "__main__":
    main() 