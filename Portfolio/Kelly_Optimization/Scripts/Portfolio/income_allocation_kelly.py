import numpy as np
from scipy.optimize import minimize

class IncomeAllocationKelly:
    def __init__(self):
        """
        Simple Kelly calculator for income allocation between:
        1. Bitcoin High-Yield Reserve Strategy (40% reserves, 60% Bitcoin)
        2. S&P 500 Strategy
        """
        
        # Your Bitcoin strategy (from your results)
        self.btc_strategy = {
            'name': 'Bitcoin High-Yield Reserve Strategy',
            'expected_return': 0.0299,  # 2.99% CAGR
            'volatility': 0.41,         # 41% volatility
            'description': '40% high-yield savings, 60% tactical Bitcoin'
        }
        
        # S&P 500 strategy
        self.sp500_strategy = {
            'name': 'S&P 500 Strategy',
            'expected_return': 0.0778,  # 7.78% from your model
            'volatility': 0.16,         # 16% volatility
            'description': '100% S&P 500 index fund'
        }
        
        # Risk-free rate (high-yield savings)
        self.risk_free_rate = 0.05  # 5%
        
        # Estimated correlation between strategies
        self.correlation = 0.3  # Bitcoin and S&P 500 moderate correlation
        
        # Calculate excess returns
        self.btc_excess = self.btc_strategy['expected_return'] - self.risk_free_rate
        self.sp500_excess = self.sp500_strategy['expected_return'] - self.risk_free_rate
        
        print(f"Bitcoin Strategy Excess Return: {self.btc_excess:.2%}")
        print(f"S&P 500 Strategy Excess Return: {self.sp500_excess:.2%}")
        print()
    
    def kelly_two_asset_allocation(self, allow_leverage=False):
        """
        Calculate Kelly-optimal allocation between Bitcoin strategy and S&P 500
        """
        
        # Parameters
        mu1 = self.btc_excess  # Bitcoin strategy excess return
        mu2 = self.sp500_excess  # S&P 500 excess return
        sigma1 = self.btc_strategy['volatility']
        sigma2 = self.sp500_strategy['volatility']
        
        # Covariance matrix
        var1 = sigma1 ** 2
        var2 = sigma2 ** 2
        cov12 = self.correlation * sigma1 * sigma2
        
        # Kelly optimal weights (analytical solution)
        denominator = var1 * var2 - cov12 ** 2
        
        if denominator == 0:
            return 0.5, 0.5  # Equal weights if singular
        
        w1 = (mu1 * var2 - mu2 * cov12) / denominator  # Bitcoin strategy weight
        w2 = (mu2 * var1 - mu1 * cov12) / denominator  # S&P 500 weight
        
        # Handle no-leverage constraint
        if not allow_leverage:
            total_weight = w1 + w2
            if total_weight > 1.0:
                w1 = w1 / total_weight
                w2 = w2 / total_weight
            
            # Ensure non-negative weights
            w1 = max(0, w1)
            w2 = max(0, w2)
            
            # Renormalize if needed
            total = w1 + w2
            if total > 1.0:
                w1 = w1 / total
                w2 = w2 / total
        
        return w1, w2
    
    def calculate_portfolio_metrics(self, w_btc, w_sp500):
        """Calculate portfolio return, volatility, and Sharpe ratio"""
        
        w_cash = 1 - w_btc - w_sp500
        
        # Portfolio return
        portfolio_return = (w_btc * self.btc_strategy['expected_return'] + 
                          w_sp500 * self.sp500_strategy['expected_return'] + 
                          w_cash * self.risk_free_rate)
        
        # Portfolio variance
        covariance = self.correlation * self.btc_strategy['volatility'] * self.sp500_strategy['volatility']
        portfolio_variance = (w_btc**2 * self.btc_strategy['volatility']**2 + 
                            w_sp500**2 * self.sp500_strategy['volatility']**2 + 
                            2 * w_btc * w_sp500 * covariance)
        
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def fractional_kelly_analysis(self, fractions=[0.25, 0.5, 0.75, 1.0]):
        """Analyze fractional Kelly allocations"""
        
        # Get full Kelly allocation
        w_btc_full, w_sp500_full = self.kelly_two_asset_allocation()
        
        results = {}
        
        for fraction in fractions:
            w_btc = w_btc_full * fraction
            w_sp500 = w_sp500_full * fraction
            w_cash = 1 - w_btc - w_sp500
            
            portfolio_return, portfolio_vol, sharpe = self.calculate_portfolio_metrics(w_btc, w_sp500)
            
            results[f'{fraction:.0%}_kelly'] = {
                'btc_allocation': w_btc,
                'sp500_allocation': w_sp500,
                'cash_allocation': w_cash,
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe
            }
        
        return results
    
    def monthly_income_breakdown(self, monthly_income=1000):
        """Show how to allocate monthly income"""
        
        print(f"📊 MONTHLY INCOME ALLOCATION (${monthly_income:,})")
        print("=" * 50)
        
        # Get Kelly allocation
        w_btc, w_sp500 = self.kelly_two_asset_allocation()
        w_cash = 1 - w_btc - w_sp500
        
        # Calculate dollar amounts
        btc_dollars = w_btc * monthly_income
        sp500_dollars = w_sp500 * monthly_income
        cash_dollars = w_cash * monthly_income
        
        print(f"Kelly Optimal Allocation:")
        print(f"  Bitcoin Strategy: ${btc_dollars:.0f} ({w_btc:.1%})")
        print(f"  S&P 500: ${sp500_dollars:.0f} ({w_sp500:.1%})")
        print(f"  High-Yield Savings: ${cash_dollars:.0f} ({w_cash:.1%})")
        print()
        
        # Show what happens inside the Bitcoin strategy
        if btc_dollars > 0:
            btc_to_reserves = btc_dollars * 0.4  # 40% to reserves
            btc_to_trading = btc_dollars * 0.6   # 60% to Bitcoin buying
            
            print(f"Inside Bitcoin Strategy (${btc_dollars:.0f}):")
            print(f"  → High-Yield Savings: ${btc_to_reserves:.0f} (40%)")
            print(f"  → Bitcoin Trading: ${btc_to_trading:.0f} (60%)")
            print()
        
        # Show fractional Kelly options
        print("🎯 FRACTIONAL KELLY OPTIONS:")
        print("-" * 30)
        
        fractional_results = self.fractional_kelly_analysis()
        
        for fraction_name, data in fractional_results.items():
            btc_frac_dollars = data['btc_allocation'] * monthly_income
            sp500_frac_dollars = data['sp500_allocation'] * monthly_income
            cash_frac_dollars = data['cash_allocation'] * monthly_income
            
            print(f"{fraction_name}:")
            print(f"  Bitcoin Strategy: ${btc_frac_dollars:.0f} ({data['btc_allocation']:.1%})")
            print(f"  S&P 500: ${sp500_frac_dollars:.0f} ({data['sp500_allocation']:.1%})")
            print(f"  High-Yield Savings: ${cash_frac_dollars:.0f} ({data['cash_allocation']:.1%})")
            print(f"  Expected Return: {data['expected_return']:.2%}")
            print(f"  Volatility: {data['volatility']:.2%}")
            print(f"  Sharpe Ratio: {data['sharpe_ratio']:.3f}")
            print()
    
    def generate_income_allocation_report(self):
        """Generate comprehensive income allocation report"""
        
        print("INCOME ALLOCATION KELLY CRITERION")
        print("=" * 50)
        print("Question: How much of monthly income goes to each strategy?")
        print()
        
        # Show strategy details
        print("📋 STRATEGY DETAILS:")
        print("-" * 30)
        print(f"Bitcoin Strategy:")
        print(f"  Expected Return: {self.btc_strategy['expected_return']:.2%}")
        print(f"  Volatility: {self.btc_strategy['volatility']:.2%}")
        print(f"  Excess Return: {self.btc_excess:.2%}")
        print(f"  Description: {self.btc_strategy['description']}")
        print()
        
        print(f"S&P 500 Strategy:")
        print(f"  Expected Return: {self.sp500_strategy['expected_return']:.2%}")
        print(f"  Volatility: {self.sp500_strategy['volatility']:.2%}")
        print(f"  Excess Return: {self.sp500_excess:.2%}")
        print(f"  Description: {self.sp500_strategy['description']}")
        print()
        
        # Kelly calculation
        w_btc, w_sp500 = self.kelly_two_asset_allocation()
        w_cash = 1 - w_btc - w_sp500
        
        portfolio_return, portfolio_vol, sharpe = self.calculate_portfolio_metrics(w_btc, w_sp500)
        
        print("🏆 KELLY OPTIMAL ALLOCATION:")
        print("-" * 30)
        print(f"Bitcoin Strategy: {w_btc:.1%}")
        print(f"S&P 500: {w_sp500:.1%}")
        print(f"High-Yield Savings: {w_cash:.1%}")
        print()
        print(f"Expected Portfolio Return: {portfolio_return:.2%}")
        print(f"Portfolio Volatility: {portfolio_vol:.2%}")
        print(f"Sharpe Ratio: {sharpe:.3f}")
        print()
        
        # Monthly breakdown
        self.monthly_income_breakdown(1000)
        
        # Recommendations
        print("💡 RECOMMENDATIONS:")
        print("-" * 30)
        
        if w_btc > 0.5:
            print("🚨 HIGH BITCOIN ALLOCATION - Consider fractional Kelly")
            print("   Recommended: 50% Kelly to reduce risk")
        elif w_btc > 0.3:
            print("⚖️ MODERATE BITCOIN ALLOCATION - Well balanced")
            print("   Can use full Kelly or 75% Kelly")
        else:
            print("🛡️ CONSERVATIVE ALLOCATION - Low Bitcoin exposure")
            print("   Full Kelly allocation is reasonable")
        
        print()
        print("🎯 IMPLEMENTATION:")
        print("   1. Set up automatic investments to each strategy")
        print("   2. Rebalance quarterly to maintain target allocation")
        print("   3. Monitor strategy performance and adjust if needed")
        print("   4. Consider fractional Kelly during volatile periods")

def main():
    """Run income allocation Kelly analysis"""
    
    calculator = IncomeAllocationKelly()
    calculator.generate_income_allocation_report()

if __name__ == "__main__":
    main() 