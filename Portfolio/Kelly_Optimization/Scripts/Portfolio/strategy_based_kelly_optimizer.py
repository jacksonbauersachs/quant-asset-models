import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class StrategyBasedKellyOptimizer:
    def __init__(self):
        """
        Initialize with your actual investment strategies as the 'assets'
        No leverage - only allocate between strategies, with remainder in high-yield savings
        """
        
        # Risk-free rate = High-yield savings (your baseline safe option)
        self.risk_free_rate = 0.05  # 5% high-yield savings
        
        # Your actual investment strategies (from your results)
        self.strategies = {
            'btc_high_yield_reserve': {
                'name': 'Bitcoin High-Yield Reserve Strategy',
                'expected_return': 0.0299,     # 2.99% CAGR from your results
                'volatility': 0.41,            # Estimated from your volatility data
                'description': '40% reserves in 5% savings, multi-tier Bitcoin buying',
                'median_return_5yr': 69532,    # From your simulation
                'improvement_vs_baseline': 0.254  # +25.4% vs baseline
            },
            'btc_cash_reserve': {
                'name': 'Bitcoin Cash Reserve Strategy', 
                'expected_return': 0.0135,     # 1.35% CAGR from your results
                'volatility': 0.43,            # Estimated from your volatility data
                'description': '10% cash reserves, multi-tier Bitcoin buying',
                'median_return_5yr': 64161,    # From your simulation
                'improvement_vs_baseline': 0.162  # +16.2% vs baseline
            },
            'btc_sp500_tactical': {
                'name': 'Bitcoin/S&P 500 Tactical Allocation',
                'expected_return': 0.0021,     # 0.21% CAGR from your results
                'volatility': 0.27,            # Lower volatility from diversification
                'description': '20% Bitcoin base, tactical rebalancing with S&P 500',
                'median_return_5yr': 60621,    # From your simulation
                'improvement_vs_baseline': 0.108  # +10.8% vs baseline
            },
            'sp500_pure': {
                'name': 'Pure S&P 500 Strategy',
                'expected_return': 0.0778,     # 7.78% from your growth model
                'volatility': 0.16,            # Historical S&P 500 volatility
                'description': '100% S&P 500 index fund allocation',
                'median_return_5yr': None,     # Not simulated in your results
                'improvement_vs_baseline': None
            },
            'btc_variance_trading': {
                'name': 'Bitcoin Variance Trading Strategy',
                'expected_return': 0.15,       # Estimated from 234% ROI over time period
                'volatility': 0.50,            # High volatility from active trading
                'description': '70% reserves, buy/sell based on growth model deviations',
                'median_return_5yr': None,     # From backtest, not forward simulation
                'improvement_vs_baseline': 2.34  # 234% ROI
            }
        }
        
        # Calculate excess returns over your risk-free rate (high-yield savings)
        for strategy in self.strategies.values():
            strategy['excess_return'] = strategy['expected_return'] - self.risk_free_rate
            strategy['sharpe_ratio'] = strategy['excess_return'] / strategy['volatility']
    
    def kelly_multi_strategy_no_leverage(self, strategy_subset=None, max_allocation=1.0):
        """
        Calculate Kelly-optimal allocation across strategies with no leverage
        Remainder goes to high-yield savings (risk-free asset)
        """
        if strategy_subset is None:
            strategy_subset = list(self.strategies.keys())
        
        n_strategies = len(strategy_subset)
        strategies = [self.strategies[s] for s in strategy_subset]
        
        # Excess returns over risk-free rate
        excess_returns = np.array([s['excess_return'] for s in strategies])
        
        # Volatilities
        volatilities = np.array([s['volatility'] for s in strategies])
        
        # Create correlation matrix (estimated based on strategy types)
        correlation_matrix = np.eye(n_strategies)
        for i in range(n_strategies):
            for j in range(i+1, n_strategies):
                # Correlation based on strategy similarity
                strategy_i = strategy_subset[i]
                strategy_j = strategy_subset[j]
                
                if 'btc' in strategy_i and 'btc' in strategy_j:
                    # Bitcoin strategies are highly correlated
                    correlation_matrix[i,j] = correlation_matrix[j,i] = 0.8
                elif 'btc' in strategy_i and 'sp500' in strategy_j:
                    # Bitcoin and S&P 500 moderate correlation
                    correlation_matrix[i,j] = correlation_matrix[j,i] = 0.3
                elif 'tactical' in strategy_i and 'tactical' in strategy_j:
                    # Tactical strategies moderately correlated
                    correlation_matrix[i,j] = correlation_matrix[j,i] = 0.6
                else:
                    # Default moderate correlation
                    correlation_matrix[i,j] = correlation_matrix[j,i] = 0.4
        
        # Covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        
        def kelly_objective(weights):
            """
            Kelly objective: maximize log-expected return
            E[log(1 + r)] ≈ μ - σ²/2 for portfolio
            """
            # Portfolio excess return
            portfolio_excess_return = np.dot(weights, excess_returns)
            
            # Portfolio variance
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            
            # Kelly objective (maximize this)
            kelly_value = portfolio_excess_return - 0.5 * portfolio_variance
            
            return -kelly_value  # Minimize negative
        
        # Constraints: weights sum to <= max_allocation, all weights >= 0
        constraints = [
            {'type': 'ineq', 'fun': lambda w: max_allocation - np.sum(w)},  # Sum <= max_allocation
        ]
        
        # Bounds: each weight between 0 and max_allocation
        bounds = [(0, max_allocation) for _ in range(n_strategies)]
        
        # Initial guess
        x0 = np.ones(n_strategies) * (max_allocation / n_strategies / 2)
        
        # Optimize
        result = minimize(
            kelly_objective,
            x0,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds
        )
        
        optimal_weights = result.x
        cash_allocation = max_allocation - np.sum(optimal_weights)
        
        return optimal_weights, cash_allocation, strategy_subset
    
    def calculate_portfolio_metrics(self, weights, cash_allocation, strategy_subset):
        """Calculate portfolio return, volatility, and Sharpe ratio"""
        strategies = [self.strategies[s] for s in strategy_subset]
        
        # Portfolio return
        strategy_returns = np.array([s['expected_return'] for s in strategies])
        portfolio_return = np.dot(weights, strategy_returns) + cash_allocation * self.risk_free_rate
        
        # Portfolio volatility (cash has zero volatility)
        volatilities = np.array([s['volatility'] for s in strategies])
        
        # Simplified correlation matrix for volatility calculation
        n_strategies = len(strategy_subset)
        correlation_matrix = np.eye(n_strategies)
        for i in range(n_strategies):
            for j in range(i+1, n_strategies):
                strategy_i = strategy_subset[i]
                strategy_j = strategy_subset[j]
                
                if 'btc' in strategy_i and 'btc' in strategy_j:
                    correlation_matrix[i,j] = correlation_matrix[j,i] = 0.8
                elif 'btc' in strategy_i and 'sp500' in strategy_j:
                    correlation_matrix[i,j] = correlation_matrix[j,i] = 0.3
                else:
                    correlation_matrix[i,j] = correlation_matrix[j,i] = 0.4
        
        # Portfolio variance
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def analyze_strategy_combinations(self):
        """Analyze different combinations of your strategies"""
        results = {}
        
        # 1. All strategies
        weights, cash, strategies = self.kelly_multi_strategy_no_leverage()
        portfolio_return, portfolio_vol, sharpe = self.calculate_portfolio_metrics(weights, cash, strategies)
        results['all_strategies'] = {
            'weights': weights,
            'cash_allocation': cash,
            'strategies': strategies,
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe
        }
        
        # 2. Bitcoin strategies only
        btc_strategies = ['btc_high_yield_reserve', 'btc_cash_reserve', 'btc_variance_trading']
        weights, cash, strategies = self.kelly_multi_strategy_no_leverage(btc_strategies)
        portfolio_return, portfolio_vol, sharpe = self.calculate_portfolio_metrics(weights, cash, strategies)
        results['btc_strategies'] = {
            'weights': weights,
            'cash_allocation': cash,
            'strategies': strategies,
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe
        }
        
        # 3. Conservative strategies (lower volatility)
        conservative_strategies = ['btc_sp500_tactical', 'sp500_pure']
        weights, cash, strategies = self.kelly_multi_strategy_no_leverage(conservative_strategies)
        portfolio_return, portfolio_vol, sharpe = self.calculate_portfolio_metrics(weights, cash, strategies)
        results['conservative'] = {
            'weights': weights,
            'cash_allocation': cash,
            'strategies': strategies,
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe
        }
        
        # 4. Top performers (best Sharpe ratios)
        top_strategies = ['sp500_pure', 'btc_sp500_tactical', 'btc_high_yield_reserve']
        weights, cash, strategies = self.kelly_multi_strategy_no_leverage(top_strategies)
        portfolio_return, portfolio_vol, sharpe = self.calculate_portfolio_metrics(weights, cash, strategies)
        results['top_performers'] = {
            'weights': weights,
            'cash_allocation': cash,
            'strategies': strategies,
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe
        }
        
        # 5. High-yield reserve vs S&P 500 (your key question)
        key_strategies = ['btc_high_yield_reserve', 'sp500_pure']
        weights, cash, strategies = self.kelly_multi_strategy_no_leverage(key_strategies)
        portfolio_return, portfolio_vol, sharpe = self.calculate_portfolio_metrics(weights, cash, strategies)
        results['btc_reserve_vs_sp500'] = {
            'weights': weights,
            'cash_allocation': cash,
            'strategies': strategies,
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe
        }
        
        return results
    
    def fractional_kelly_analysis(self, strategy_subset=None, fractions=[0.25, 0.5, 0.75, 1.0]):
        """Analyze fractional Kelly allocations to reduce risk"""
        if strategy_subset is None:
            strategy_subset = ['btc_high_yield_reserve', 'sp500_pure']
        
        results = {}
        
        # Get full Kelly allocation
        full_weights, full_cash, strategies = self.kelly_multi_strategy_no_leverage(strategy_subset)
        
        for fraction in fractions:
            # Apply fractional Kelly
            frac_weights = full_weights * fraction
            frac_cash = 1 - np.sum(frac_weights)  # Remainder goes to high-yield savings
            
            # Calculate metrics
            portfolio_return, portfolio_vol, sharpe = self.calculate_portfolio_metrics(frac_weights, frac_cash, strategies)
            
            results[f'{fraction:.0%}_kelly'] = {
                'weights': frac_weights,
                'cash_allocation': frac_cash,
                'strategies': strategies,
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe,
                'fraction': fraction
            }
        
        return results
    
    def generate_strategy_kelly_report(self):
        """Generate comprehensive report using your actual strategies"""
        
        print("STRATEGY-BASED KELLY CRITERION OPTIMIZATION")
        print("=" * 70)
        print(f"Risk-Free Rate: {self.risk_free_rate:.1%} (High-Yield Savings)")
        print("No Leverage - Remainder allocated to high-yield savings")
        print()
        
        # 1. Individual Strategy Analysis
        print("📊 INDIVIDUAL STRATEGY ANALYSIS:")
        print("-" * 50)
        
        strategies_sorted = sorted(
            self.strategies.items(),
            key=lambda x: x[1]['sharpe_ratio'],
            reverse=True
        )
        
        for i, (key, strategy) in enumerate(strategies_sorted, 1):
            print(f"#{i}. {strategy['name']}")
            print(f"    Expected Return: {strategy['expected_return']:.2%}")
            print(f"    Volatility: {strategy['volatility']:.2%}")
            print(f"    Excess Return: {strategy['excess_return']:.2%}")
            print(f"    Sharpe Ratio: {strategy['sharpe_ratio']:.3f}")
            print(f"    Description: {strategy['description']}")
            print()
        
        # 2. Key Strategy Combination (Your Main Question)
        print("🎯 KEY QUESTION: BTC HIGH-YIELD RESERVE vs S&P 500")
        print("-" * 50)
        
        key_results = self.analyze_strategy_combinations()
        key_combo = key_results['btc_reserve_vs_sp500']
        
        print("Kelly-Optimal Allocation:")
        for i, strategy_name in enumerate(key_combo['strategies']):
            weight = key_combo['weights'][i]
            strategy_display = self.strategies[strategy_name]['name']
            print(f"  {strategy_display}: {weight:.1%}")
        
        print(f"  High-Yield Savings: {key_combo['cash_allocation']:.1%}")
        print(f"Expected Return: {key_combo['expected_return']:.2%}")
        print(f"Volatility: {key_combo['volatility']:.2%}")
        print(f"Sharpe Ratio: {key_combo['sharpe_ratio']:.3f}")
        print()
        
        # 3. Fractional Kelly Analysis
        print("💰 FRACTIONAL KELLY ANALYSIS:")
        print("-" * 50)
        
        fractional_results = self.fractional_kelly_analysis(['btc_high_yield_reserve', 'sp500_pure'])
        
        for fraction_name, data in fractional_results.items():
            print(f"{fraction_name}:")
            for i, strategy_name in enumerate(data['strategies']):
                weight = data['weights'][i]
                strategy_display = self.strategies[strategy_name]['name']
                if weight > 0.001:  # Only show if > 0.1%
                    print(f"  {strategy_display}: {weight:.1%}")
            print(f"  High-Yield Savings: {data['cash_allocation']:.1%}")
            print(f"  Return: {data['expected_return']:.2%}, Vol: {data['volatility']:.2%}, Sharpe: {data['sharpe_ratio']:.3f}")
            print()
        
        # 4. All Strategy Combinations
        print("🔄 ALL STRATEGY COMBINATIONS:")
        print("-" * 50)
        
        all_results = self.analyze_strategy_combinations()
        
        for combo_name, combo_data in all_results.items():
            print(f"{combo_name.upper().replace('_', ' ')}:")
            print(f"  Expected Return: {combo_data['expected_return']:.2%}")
            print(f"  Volatility: {combo_data['volatility']:.2%}")
            print(f"  Sharpe Ratio: {combo_data['sharpe_ratio']:.3f}")
            print("  Allocation:")
            for i, strategy_name in enumerate(combo_data['strategies']):
                weight = combo_data['weights'][i]
                if weight > 0.01:  # Only show allocations > 1%
                    strategy_display = self.strategies[strategy_name]['name']
                    print(f"    {strategy_display}: {weight:.1%}")
            if combo_data['cash_allocation'] > 0.01:
                print(f"    High-Yield Savings: {combo_data['cash_allocation']:.1%}")
            print()
        
        # 5. Final Recommendations
        print("🎯 FINAL RECOMMENDATIONS:")
        print("-" * 50)
        
        # Find best Sharpe ratio combination
        best_combo = max(all_results.items(), key=lambda x: x[1]['sharpe_ratio'])
        
        print(f"🏆 BEST RISK-ADJUSTED STRATEGY: {best_combo[0].upper().replace('_', ' ')}")
        print(f"   Expected Return: {best_combo[1]['expected_return']:.2%}")
        print(f"   Volatility: {best_combo[1]['volatility']:.2%}")
        print(f"   Sharpe Ratio: {best_combo[1]['sharpe_ratio']:.3f}")
        print()
        print("   Optimal Allocation:")
        for i, strategy_name in enumerate(best_combo[1]['strategies']):
            weight = best_combo[1]['weights'][i]
            if weight > 0.01:
                strategy_display = self.strategies[strategy_name]['name']
                print(f"   • {weight:.1%} → {strategy_display}")
        if best_combo[1]['cash_allocation'] > 0.01:
            print(f"   • {best_combo[1]['cash_allocation']:.1%} → High-Yield Savings")
        
        print()
        print("💡 KEY INSIGHTS:")
        print("   • Your high-yield reserve strategy changes the risk/return profile")
        print("   • Kelly criterion accounts for the actual strategy returns, not raw asset returns")
        print("   • Fractional Kelly reduces risk while maintaining good returns")
        print("   • High-yield savings as 'cash' reduces opportunity cost significantly")

def main():
    """Run strategy-based Kelly optimization"""
    
    optimizer = StrategyBasedKellyOptimizer()
    optimizer.generate_strategy_kelly_report()

if __name__ == "__main__":
    main() 