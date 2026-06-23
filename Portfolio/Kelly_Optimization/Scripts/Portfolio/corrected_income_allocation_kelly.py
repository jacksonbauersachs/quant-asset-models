import numpy as np
from scipy.optimize import minimize

class CorrectedIncomeAllocationKelly:
    def __init__(self):
        """
        Corrected Kelly calculator comparing:
        1. Bitcoin High-Yield Reserve Strategy (your improved Bitcoin approach)
        2. S&P 500 Strategy
        
        NO risk-free rate - just comparing two investment strategies
        """
        
        # Your improved Bitcoin strategy (from your results)
        self.btc_strategy = {
            'name': 'Bitcoin High-Yield Reserve Strategy',
            'expected_return': 0.0299,  # 2.99% CAGR from your results
            'volatility': 0.41,         # 41% volatility
            'description': '40% high-yield savings, 60% tactical Bitcoin - BEATS basic DCA',
            'improvement_vs_dca': 0.254  # +25.4% improvement vs basic DCA
        }
        
        # S&P 500 strategy
        self.sp500_strategy = {
            'name': 'S&P 500 Strategy',
            'expected_return': 0.0778,  # 7.78% from your model
            'volatility': 0.16,         # 16% volatility
            'description': '100% S&P 500 index fund'
        }
        
        # Original raw Bitcoin (for comparison)
        self.raw_bitcoin = {
            'name': 'Raw Bitcoin',
            'expected_return': 0.2131,  # 21.31% from your growth model
            'volatility': 0.60,         # 60% volatility
            'description': 'Direct Bitcoin investment'
        }
        
        # Estimated correlation between Bitcoin strategy and S&P 500
        self.correlation = 0.3  # Moderate correlation
        
        print("🔄 STRATEGY COMPARISON:")
        print(f"Raw Bitcoin: {self.raw_bitcoin['expected_return']:.2%} return, {self.raw_bitcoin['volatility']:.1%} volatility")
        print(f"Your Bitcoin Strategy: {self.btc_strategy['expected_return']:.2%} return, {self.btc_strategy['volatility']:.1%} volatility")
        print(f"S&P 500: {self.sp500_strategy['expected_return']:.2%} return, {self.sp500_strategy['volatility']:.1%} volatility")
        print()
    
    def kelly_two_strategy_allocation(self):
        """
        Calculate Kelly-optimal allocation between your Bitcoin strategy and S&P 500
        This is the CORRECT approach - no risk-free rate needed
        """
        
        # Strategy parameters
        mu1 = self.btc_strategy['expected_return']
        mu2 = self.sp500_strategy['expected_return']
        sigma1 = self.btc_strategy['volatility']
        sigma2 = self.sp500_strategy['volatility']
        
        # Covariance matrix
        var1 = sigma1 ** 2
        var2 = sigma2 ** 2
        cov12 = self.correlation * sigma1 * sigma2
        
        # Kelly optimal weights for two risky assets (no risk-free asset)
        # This maximizes log(expected return) of the portfolio
        
        def kelly_objective(weights):
            w1, w2 = weights
            
            # Portfolio return
            portfolio_return = w1 * mu1 + w2 * mu2
            
            # Portfolio variance
            portfolio_variance = w1**2 * var1 + w2**2 * var2 + 2 * w1 * w2 * cov12
            
            # Kelly objective (maximize log expected return)
            # For small returns: log(1 + r) ≈ r - r²/2
            kelly_value = portfolio_return - 0.5 * portfolio_variance
            
            return -kelly_value  # Minimize negative
        
        # Constraints: weights sum to 1, both weights >= 0
        constraints = [
            {'type': 'eq', 'fun': lambda w: w[0] + w[1] - 1},  # Sum to 1
            {'type': 'ineq', 'fun': lambda w: w[0]},           # w1 >= 0
            {'type': 'ineq', 'fun': lambda w: w[1]}            # w2 >= 0
        ]
        
        # Optimize
        result = minimize(
            kelly_objective,
            [0.5, 0.5],  # Initial guess
            method='SLSQP',
            constraints=constraints,
            bounds=[(0, 1), (0, 1)]
        )
        
        return result.x[0], result.x[1]  # Bitcoin weight, S&P 500 weight
    
    def compare_to_original_kelly(self):
        """
        Compare to original Kelly allocation using raw Bitcoin
        """
        
        # Original Kelly with raw Bitcoin vs S&P 500
        mu1_raw = self.raw_bitcoin['expected_return']
        mu2 = self.sp500_strategy['expected_return']
        sigma1_raw = self.raw_bitcoin['volatility']
        sigma2 = self.sp500_strategy['volatility']
        
        # Covariance
        var1_raw = sigma1_raw ** 2
        var2 = sigma2 ** 2
        cov12_raw = self.correlation * sigma1_raw * sigma2
        
        def kelly_objective_raw(weights):
            w1, w2 = weights
            portfolio_return = w1 * mu1_raw + w2 * mu2
            portfolio_variance = w1**2 * var1_raw + w2**2 * var2 + 2 * w1 * w2 * cov12_raw
            kelly_value = portfolio_return - 0.5 * portfolio_variance
            return -kelly_value
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: w[0] + w[1] - 1},
            {'type': 'ineq', 'fun': lambda w: w[0]},
            {'type': 'ineq', 'fun': lambda w: w[1]}
        ]
        
        result_raw = minimize(
            kelly_objective_raw,
            [0.5, 0.5],
            method='SLSQP',
            constraints=constraints,
            bounds=[(0, 1), (0, 1)]
        )
        
        return result_raw.x[0], result_raw.x[1]  # Raw Bitcoin weight, S&P 500 weight
    
    def calculate_portfolio_metrics(self, w_btc, w_sp500, strategy_type='improved'):
        """Calculate portfolio return and volatility"""
        
        if strategy_type == 'improved':
            btc_return = self.btc_strategy['expected_return']
            btc_vol = self.btc_strategy['volatility']
        else:
            btc_return = self.raw_bitcoin['expected_return']
            btc_vol = self.raw_bitcoin['volatility']
        
        sp500_return = self.sp500_strategy['expected_return']
        sp500_vol = self.sp500_strategy['volatility']
        
        # Portfolio return
        portfolio_return = w_btc * btc_return + w_sp500 * sp500_return
        
        # Portfolio variance
        covariance = self.correlation * btc_vol * sp500_vol
        portfolio_variance = (w_btc**2 * btc_vol**2 + 
                            w_sp500**2 * sp500_vol**2 + 
                            2 * w_btc * w_sp500 * covariance)
        
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        return portfolio_return, portfolio_volatility
    
    def monthly_income_breakdown(self, monthly_income=1000):
        """Show how to allocate monthly income"""
        
        print(f"📊 MONTHLY INCOME ALLOCATION (${monthly_income:,})")
        print("=" * 50)
        
        # Get improved Kelly allocation
        w_btc_improved, w_sp500_improved = self.kelly_two_strategy_allocation()
        
        # Get original Kelly allocation for comparison
        w_btc_raw, w_sp500_raw = self.compare_to_original_kelly()
        
        # Calculate dollar amounts for improved strategy
        btc_dollars = w_btc_improved * monthly_income
        sp500_dollars = w_sp500_improved * monthly_income
        
        print(f"🏆 IMPROVED KELLY ALLOCATION:")
        print(f"  Bitcoin Strategy: ${btc_dollars:.0f} ({w_btc_improved:.1%})")
        print(f"  S&P 500: ${sp500_dollars:.0f} ({w_sp500_improved:.1%})")
        print()
        
        # Show what happens inside the Bitcoin strategy
        if btc_dollars > 0:
            btc_to_reserves = btc_dollars * 0.4  # 40% to high-yield savings
            btc_to_trading = btc_dollars * 0.6   # 60% to Bitcoin buying
            
            print(f"Inside Bitcoin Strategy (${btc_dollars:.0f}):")
            print(f"  → High-Yield Savings: ${btc_to_reserves:.0f} (40%)")
            print(f"  → Bitcoin Trading: ${btc_to_trading:.0f} (60%)")
            print()
        
        # Compare to original
        print(f"📈 COMPARISON TO ORIGINAL KELLY:")
        print(f"  Original (Raw Bitcoin): {w_btc_raw:.1%} BTC, {w_sp500_raw:.1%} S&P")
        print(f"  Improved (Your Strategy): {w_btc_improved:.1%} BTC, {w_sp500_improved:.1%} S&P")
        print()
        
        if w_btc_improved > w_btc_raw:
            print("✅ YOUR IMPROVED STRATEGY GETS MORE ALLOCATION!")
        elif w_btc_improved < w_btc_raw:
            print("❌ Your improved strategy gets less allocation")
        else:
            print("➖ Same allocation as original")
    
    def generate_corrected_report(self):
        """Generate the corrected analysis"""
        
        print("CORRECTED INCOME ALLOCATION ANALYSIS")
        print("=" * 50)
        print("Comparing: Your Bitcoin Strategy vs S&P 500")
        print("No risk-free rate - just optimal allocation between strategies")
        print()
        
        # Get allocations
        w_btc_improved, w_sp500_improved = self.kelly_two_strategy_allocation()
        w_btc_raw, w_sp500_raw = self.compare_to_original_kelly()
        
        # Calculate portfolio metrics
        improved_return, improved_vol = self.calculate_portfolio_metrics(w_btc_improved, w_sp500_improved, 'improved')
        raw_return, raw_vol = self.calculate_portfolio_metrics(w_btc_raw, w_sp500_raw, 'raw')
        
        print("🏆 KELLY OPTIMAL ALLOCATIONS:")
        print("-" * 40)
        print(f"With Your Improved Bitcoin Strategy:")
        print(f"  Bitcoin Strategy: {w_btc_improved:.1%}")
        print(f"  S&P 500: {w_sp500_improved:.1%}")
        print(f"  Expected Return: {improved_return:.2%}")
        print(f"  Volatility: {improved_vol:.2%}")
        print()
        
        print(f"With Raw Bitcoin (Original):")
        print(f"  Raw Bitcoin: {w_btc_raw:.1%}")
        print(f"  S&P 500: {w_sp500_raw:.1%}")
        print(f"  Expected Return: {raw_return:.2%}")
        print(f"  Volatility: {raw_vol:.2%}")
        print()
        
        # Analysis
        print("🔍 ANALYSIS:")
        print("-" * 40)
        
        allocation_change = w_btc_improved - w_btc_raw
        
        if allocation_change > 0:
            print(f"✅ Your improved strategy gets {allocation_change:.1%} MORE allocation!")
            print(f"   This makes sense because it has better risk-adjusted returns")
        elif allocation_change < 0:
            print(f"❌ Your improved strategy gets {abs(allocation_change):.1%} LESS allocation")
            print(f"   The lower volatility doesn't compensate for the lower returns")
        else:
            print("➖ Same allocation - strategies have similar risk-adjusted returns")
        
        print()
        self.monthly_income_breakdown(1000)
        
        print("💡 KEY INSIGHTS:")
        print("-" * 40)
        print("1. Kelly criterion compares risk-adjusted returns between strategies")
        print("2. Your improved Bitcoin strategy should get different allocation than raw Bitcoin")
        print("3. The math accounts for both return AND risk differences")
        print("4. No borrowing/leverage - just optimal allocation of your income")

def main():
    """Run corrected income allocation analysis"""
    
    calculator = CorrectedIncomeAllocationKelly()
    calculator.generate_corrected_report()

if __name__ == "__main__":
    main() 