import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class BitcoinSP500AllocationOptimizer:
    def __init__(self):
        """Initialize with market parameters from your models"""
        
        # Base parameters from your growth models
        self.bitcoin_params = {
            'base_return': 0.2131,      # 21.31% from your 10-year model
            'volatility': 0.60,         # 60% volatility
            'sharpe_ratio': None
        }
        
        self.sp500_params = {
            'base_return': 0.0778,      # 7.78% from your model  
            'volatility': 0.16,         # 16% volatility
            'sharpe_ratio': None
        }
        
        # Market conditions
        self.risk_free_rate = 0.03      # 3% risk-free rate
        self.correlation = 0.3          # BTC-S&P 500 correlation
        
        # Calculate Sharpe ratios
        self.bitcoin_params['sharpe_ratio'] = (self.bitcoin_params['base_return'] - self.risk_free_rate) / self.bitcoin_params['volatility']
        self.sp500_params['sharpe_ratio'] = (self.sp500_params['base_return'] - self.risk_free_rate) / self.sp500_params['volatility']
        
        # Strategy variations (from your results)
        self.strategy_variations = {
            'conservative': {
                'bitcoin_return': 0.0021,   # Tactical allocation return
                'bitcoin_volatility': 0.27,  # Lower volatility from tactical
                'description': 'Tactical allocation with trend following'
            },
            'moderate': {
                'bitcoin_return': 0.0135,   # Cash reserve strategy
                'bitcoin_volatility': 0.43,  # Moderate volatility
                'description': 'Multi-tier reserve strategy'
            },
            'aggressive': {
                'bitcoin_return': 0.0299,   # High-yield reserve strategy
                'bitcoin_volatility': 0.41,  # High volatility
                'description': 'High-yield savings reserve strategy'
            }
        }
    
    def kelly_allocation(self, bitcoin_return=None, bitcoin_vol=None, sp500_return=None, sp500_vol=None, 
                        correlation=None, allow_leverage=False):
        """
        Calculate optimal Kelly allocation between Bitcoin and S&P 500
        """
        # Use provided parameters or defaults
        btc_return = bitcoin_return or self.bitcoin_params['base_return']
        btc_vol = bitcoin_vol or self.bitcoin_params['volatility']
        sp500_return = sp500_return or self.sp500_params['base_return']
        sp500_vol = sp500_vol or self.sp500_params['volatility']
        corr = correlation or self.correlation
        
        # Excess returns
        btc_excess = btc_return - self.risk_free_rate
        sp500_excess = sp500_return - self.risk_free_rate
        
        # Covariance matrix
        btc_var = btc_vol ** 2
        sp500_var = sp500_vol ** 2
        covariance = corr * btc_vol * sp500_vol
        
        # Kelly optimal weights (analytical solution)
        denominator = btc_var * sp500_var - covariance ** 2
        
        if denominator == 0:
            return 0.5, 0.5  # Equal weights if singular
        
        w_btc = (btc_excess * sp500_var - sp500_excess * covariance) / denominator
        w_sp500 = (sp500_excess * btc_var - btc_excess * covariance) / denominator
        
        # Handle leverage constraint
        if not allow_leverage:
            total_weight = w_btc + w_sp500
            if total_weight > 1.0:
                w_btc = w_btc / total_weight
                w_sp500 = w_sp500 / total_weight
            
            # Ensure non-negative weights
            w_btc = max(0, w_btc)
            w_sp500 = max(0, w_sp500)
            
            # Renormalize if needed
            total = w_btc + w_sp500
            if total > 1.0:
                w_btc = w_btc / total
                w_sp500 = w_sp500 / total
        
        return w_btc, w_sp500
    
    def fractional_kelly(self, fraction=0.25):
        """Apply fractional Kelly to reduce risk"""
        allocations = {}
        
        for risk_level in ['conservative', 'moderate', 'aggressive']:
            # Get base allocation
            if risk_level == 'conservative':
                w_btc, w_sp500 = self.kelly_allocation(
                    bitcoin_return=self.strategy_variations[risk_level]['bitcoin_return'],
                    bitcoin_vol=self.strategy_variations[risk_level]['bitcoin_volatility']
                )
            else:
                w_btc, w_sp500 = self.kelly_allocation()
            
            # Apply fractional Kelly
            w_btc_frac = w_btc * fraction
            w_sp500_frac = w_sp500 * fraction
            w_cash = 1 - w_btc_frac - w_sp500_frac
            
            allocations[risk_level] = {
                'bitcoin': w_btc_frac,
                'sp500': w_sp500_frac,
                'cash': w_cash,
                'full_kelly_btc': w_btc,
                'full_kelly_sp500': w_sp500
            }
        
        return allocations
    
    def optimize_for_target_return(self, target_return=0.10):
        """Find allocation that achieves target return with minimum risk"""
        
        def objective(weights):
            w_btc, w_sp500 = weights
            w_cash = 1 - w_btc - w_sp500
            
            # Portfolio variance
            covariance = self.correlation * self.bitcoin_params['volatility'] * self.sp500_params['volatility']
            portfolio_variance = (w_btc**2 * self.bitcoin_params['volatility']**2 + 
                                w_sp500**2 * self.sp500_params['volatility']**2 + 
                                2 * w_btc * w_sp500 * covariance)
            
            return portfolio_variance
        
        def return_constraint(weights):
            w_btc, w_sp500 = weights
            w_cash = 1 - w_btc - w_sp500
            
            portfolio_return = (w_btc * self.bitcoin_params['base_return'] + 
                              w_sp500 * self.sp500_params['base_return'] + 
                              w_cash * self.risk_free_rate)
            
            return portfolio_return - target_return
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': return_constraint},
            {'type': 'ineq', 'fun': lambda w: 1 - w[0] - w[1]},  # w_btc + w_sp500 <= 1
            {'type': 'ineq', 'fun': lambda w: w[0]},              # w_btc >= 0
            {'type': 'ineq', 'fun': lambda w: w[1]}               # w_sp500 >= 0
        ]
        
        # Optimize
        result = minimize(
            objective,
            [0.3, 0.5],  # Initial guess
            method='SLSQP',
            constraints=constraints,
            bounds=[(0, 1), (0, 1)]
        )
        
        if result.success:
            w_btc, w_sp500 = result.x
            w_cash = 1 - w_btc - w_sp500
            portfolio_vol = np.sqrt(result.fun)
            
            return {
                'bitcoin': w_btc,
                'sp500': w_sp500,
                'cash': w_cash,
                'volatility': portfolio_vol,
                'sharpe_ratio': (target_return - self.risk_free_rate) / portfolio_vol
            }
        
        return None
    
    def analyze_risk_levels(self):
        """Analyze optimal allocations for different risk tolerance levels"""
        
        risk_levels = {
            'ultra_conservative': {'max_vol': 0.08, 'description': 'Ultra Conservative (8% max volatility)'},
            'conservative': {'max_vol': 0.12, 'description': 'Conservative (12% max volatility)'},
            'moderate': {'max_vol': 0.18, 'description': 'Moderate (18% max volatility)'},
            'aggressive': {'max_vol': 0.25, 'description': 'Aggressive (25% max volatility)'},
            'ultra_aggressive': {'max_vol': 0.35, 'description': 'Ultra Aggressive (35% max volatility)'}
        }
        
        results = {}
        
        for level, params in risk_levels.items():
            max_vol = params['max_vol']
            
            def objective(weights):
                w_btc, w_sp500 = weights
                w_cash = 1 - w_btc - w_sp500
                
                # Portfolio return
                portfolio_return = (w_btc * self.bitcoin_params['base_return'] + 
                                  w_sp500 * self.sp500_params['base_return'] + 
                                  w_cash * self.risk_free_rate)
                
                return -portfolio_return  # Maximize return
            
            def volatility_constraint(weights):
                w_btc, w_sp500 = weights
                
                covariance = self.correlation * self.bitcoin_params['volatility'] * self.sp500_params['volatility']
                portfolio_variance = (w_btc**2 * self.bitcoin_params['volatility']**2 + 
                                    w_sp500**2 * self.sp500_params['volatility']**2 + 
                                    2 * w_btc * w_sp500 * covariance)
                
                return max_vol**2 - portfolio_variance  # Volatility <= max_vol
            
            constraints = [
                {'type': 'ineq', 'fun': volatility_constraint},
                {'type': 'ineq', 'fun': lambda w: 1 - w[0] - w[1]},  # w_btc + w_sp500 <= 1
                {'type': 'ineq', 'fun': lambda w: w[0]},              # w_btc >= 0
                {'type': 'ineq', 'fun': lambda w: w[1]}               # w_sp500 >= 0
            ]
            
            result = minimize(
                objective,
                [0.2, 0.6],  # Initial guess
                method='SLSQP',
                constraints=constraints,
                bounds=[(0, 1), (0, 1)]
            )
            
            if result.success:
                w_btc, w_sp500 = result.x
                w_cash = 1 - w_btc - w_sp500
                
                # Calculate actual portfolio metrics
                portfolio_return = (w_btc * self.bitcoin_params['base_return'] + 
                                  w_sp500 * self.sp500_params['base_return'] + 
                                  w_cash * self.risk_free_rate)
                
                covariance = self.correlation * self.bitcoin_params['volatility'] * self.sp500_params['volatility']
                portfolio_variance = (w_btc**2 * self.bitcoin_params['volatility']**2 + 
                                    w_sp500**2 * self.sp500_params['volatility']**2 + 
                                    2 * w_btc * w_sp500 * covariance)
                portfolio_vol = np.sqrt(portfolio_variance)
                
                results[level] = {
                    'bitcoin': w_btc,
                    'sp500': w_sp500,
                    'cash': w_cash,
                    'expected_return': portfolio_return,
                    'volatility': portfolio_vol,
                    'sharpe_ratio': (portfolio_return - self.risk_free_rate) / portfolio_vol,
                    'description': params['description']
                }
        
        return results
    
    def generate_allocation_report(self):
        """Generate comprehensive allocation report"""
        
        print("BITCOIN vs S&P 500 ALLOCATION OPTIMIZER")
        print("=" * 70)
        print()
        
        # 1. Base Kelly Allocation
        print("📊 BASE KELLY ALLOCATION:")
        print("-" * 40)
        w_btc, w_sp500 = self.kelly_allocation()
        w_cash = 1 - w_btc - w_sp500
        
        portfolio_return = (w_btc * self.bitcoin_params['base_return'] + 
                          w_sp500 * self.sp500_params['base_return'] + 
                          w_cash * self.risk_free_rate)
        
        covariance = self.correlation * self.bitcoin_params['volatility'] * self.sp500_params['volatility']
        portfolio_variance = (w_btc**2 * self.bitcoin_params['volatility']**2 + 
                            w_sp500**2 * self.sp500_params['volatility']**2 + 
                            2 * w_btc * w_sp500 * covariance)
        portfolio_vol = np.sqrt(portfolio_variance)
        
        print(f"Bitcoin: {w_btc:.1%}")
        print(f"S&P 500: {w_sp500:.1%}")
        print(f"Cash: {w_cash:.1%}")
        print(f"Expected Return: {portfolio_return:.2%}")
        print(f"Volatility: {portfolio_vol:.2%}")
        print(f"Sharpe Ratio: {(portfolio_return - self.risk_free_rate) / portfolio_vol:.3f}")
        print()
        
        # 2. Fractional Kelly Allocations
        print("🎯 FRACTIONAL KELLY ALLOCATIONS:")
        print("-" * 40)
        
        fractions = [0.25, 0.50, 0.75, 1.00]
        
        for fraction in fractions:
            w_btc_frac = w_btc * fraction
            w_sp500_frac = w_sp500 * fraction
            w_cash_frac = 1 - w_btc_frac - w_sp500_frac
            
            portfolio_return_frac = (w_btc_frac * self.bitcoin_params['base_return'] + 
                                   w_sp500_frac * self.sp500_params['base_return'] + 
                                   w_cash_frac * self.risk_free_rate)
            
            portfolio_variance_frac = (w_btc_frac**2 * self.bitcoin_params['volatility']**2 + 
                                     w_sp500_frac**2 * self.sp500_params['volatility']**2 + 
                                     2 * w_btc_frac * w_sp500_frac * covariance)
            portfolio_vol_frac = np.sqrt(portfolio_variance_frac)
            
            print(f"{fraction:.0%} Kelly: BTC {w_btc_frac:.1%}, S&P {w_sp500_frac:.1%}, Cash {w_cash_frac:.1%}")
            print(f"   Return: {portfolio_return_frac:.2%}, Vol: {portfolio_vol_frac:.2%}, Sharpe: {(portfolio_return_frac - self.risk_free_rate) / portfolio_vol_frac:.3f}")
        
        print()
        
        # 3. Risk-Based Allocations
        print("🛡️  RISK-BASED ALLOCATIONS:")
        print("-" * 40)
        
        risk_results = self.analyze_risk_levels()
        
        for level, data in risk_results.items():
            print(f"{data['description']}:")
            print(f"   Bitcoin: {data['bitcoin']:.1%}")
            print(f"   S&P 500: {data['sp500']:.1%}")
            print(f"   Cash: {data['cash']:.1%}")
            print(f"   Expected Return: {data['expected_return']:.2%}")
            print(f"   Volatility: {data['volatility']:.2%}")
            print(f"   Sharpe Ratio: {data['sharpe_ratio']:.3f}")
            print()
        
        # 4. Target Return Allocations
        print("🎯 TARGET RETURN ALLOCATIONS:")
        print("-" * 40)
        
        target_returns = [0.06, 0.08, 0.10, 0.12, 0.15]
        
        for target in target_returns:
            result = self.optimize_for_target_return(target)
            if result:
                print(f"Target {target:.0%} Return:")
                print(f"   Bitcoin: {result['bitcoin']:.1%}")
                print(f"   S&P 500: {result['sp500']:.1%}")
                print(f"   Cash: {result['cash']:.1%}")
                print(f"   Volatility: {result['volatility']:.2%}")
                print(f"   Sharpe Ratio: {result['sharpe_ratio']:.3f}")
            else:
                print(f"Target {target:.0%} Return: Not achievable with current constraints")
            print()
        
        # 5. Recommendations
        print("💡 RECOMMENDATIONS:")
        print("-" * 40)
        
        print("🏆 OPTIMAL ALLOCATION (Kelly Criterion):")
        print(f"   • {w_btc:.1%} Bitcoin")
        print(f"   • {w_sp500:.1%} S&P 500")
        print(f"   • {w_cash:.1%} Cash")
        print(f"   • Expected Return: {portfolio_return:.2%}")
        print(f"   • Risk Level: Medium-High")
        print()
        
        # Find best risk-adjusted allocation
        best_sharpe = max(risk_results.items(), key=lambda x: x[1]['sharpe_ratio'])
        print(f"🎯 BEST RISK-ADJUSTED ({best_sharpe[1]['description']}):")
        print(f"   • {best_sharpe[1]['bitcoin']:.1%} Bitcoin")
        print(f"   • {best_sharpe[1]['sp500']:.1%} S&P 500")
        print(f"   • {best_sharpe[1]['cash']:.1%} Cash")
        print(f"   • Expected Return: {best_sharpe[1]['expected_return']:.2%}")
        print(f"   • Sharpe Ratio: {best_sharpe[1]['sharpe_ratio']:.3f}")
        print()
        
        print("📈 STRATEGY SELECTION GUIDE:")
        print("   • Conservative: 5-15% Bitcoin, focus on S&P 500")
        print("   • Moderate: 25-40% Bitcoin (Kelly optimal)")
        print("   • Aggressive: 40-60% Bitcoin, higher volatility")
        print("   • Consider fractional Kelly (25-50%) to reduce risk")

def main():
    """Run Bitcoin vs S&P 500 allocation optimization"""
    
    optimizer = BitcoinSP500AllocationOptimizer()
    optimizer.generate_allocation_report()

if __name__ == "__main__":
    main() 