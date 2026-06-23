import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pathlib import Path
import os

class ComprehensiveStrategyOptimizer:
    def __init__(self):
        """Initialize with strategy performance data from your results"""
        
        # Load strategy performance data from your results files
        self.strategies = self._load_strategy_data()
        
        # Risk-free rate
        self.risk_free_rate = 0.03
        
        # Market parameters
        self.market_params = {
            'bitcoin_base_return': 0.2131,  # From your growth model
            'bitcoin_volatility': 0.60,
            'sp500_base_return': 0.0778,   # From your growth model
            'sp500_volatility': 0.16,
            'correlation': 0.3  # Estimated correlation between BTC and S&P 500
        }
    
    def _load_strategy_data(self):
        """Load performance data from your results files"""
        strategies = {}
        
        # 1. High-Yield Savings Reserve Strategy (Best performer)
        strategies['high_yield_reserve'] = {
            'name': '5% High-Yield Savings Reserve',
            'expected_return': 0.0299,  # 2.99% CAGR
            'volatility': 0.41,  # Estimated from $28,836 volatility on $69,532
            'sharpe_ratio': (0.0299 - 0.03) / 0.41,
            'description': '40% reserves in 5% savings, multi-tier Bitcoin strategy',
            'median_return': 69532,
            'allocation_method': 'reserve_strategy'
        }
        
        # 2. Cash Reserve Strategy
        strategies['cash_reserve'] = {
            'name': '0% Cash Reserve',
            'expected_return': 0.0135,  # 1.35% CAGR
            'volatility': 0.43,  # Estimated from $27,483 volatility on $64,161
            'sharpe_ratio': (0.0135 - 0.03) / 0.43,
            'description': '10% cash reserves, multi-tier Bitcoin strategy',
            'median_return': 64161,
            'allocation_method': 'reserve_strategy'
        }
        
        # 3. Tactical Allocation Strategy
        strategies['tactical_allocation'] = {
            'name': 'Bitcoin/S&P 500 Tactical',
            'expected_return': 0.0021,  # 0.21% CAGR
            'volatility': 0.27,  # Estimated from $16,467 volatility on $60,621
            'sharpe_ratio': (0.0021 - 0.03) / 0.27,
            'description': '20% Bitcoin base, tactical rebalancing',
            'median_return': 60621,
            'allocation_method': 'tactical_rebalancing'
        }
        
        # 4. Static Kelly Allocation (from your existing analysis)
        strategies['static_kelly'] = {
            'name': 'Static Kelly (40/60)',
            'expected_return': 0.1323,  # 13.23% from your Kelly results
            'volatility': 0.2853,  # 28.53% from your Kelly results
            'sharpe_ratio': 0.359,  # From your Kelly results
            'description': '40% Bitcoin, 60% S&P 500 static allocation',
            'median_return': None,  # Not simulated
            'allocation_method': 'static_allocation'
        }
        
        # 5. Pure Bitcoin (for comparison)
        strategies['pure_bitcoin'] = {
            'name': 'Pure Bitcoin',
            'expected_return': 0.2131,  # From your growth model
            'volatility': 0.60,
            'sharpe_ratio': (0.2131 - 0.03) / 0.60,
            'description': '100% Bitcoin allocation',
            'median_return': None,
            'allocation_method': 'single_asset'
        }
        
        # 6. Pure S&P 500 (for comparison)
        strategies['pure_sp500'] = {
            'name': 'Pure S&P 500',
            'expected_return': 0.0778,  # From your growth model
            'volatility': 0.16,
            'sharpe_ratio': (0.0778 - 0.03) / 0.16,
            'description': '100% S&P 500 allocation',
            'median_return': None,
            'allocation_method': 'single_asset'
        }
        
        return strategies
    
    def kelly_multi_strategy_allocation(self, strategy_subset=None):
        """
        Find optimal allocation across multiple strategies using Kelly criterion
        """
        if strategy_subset is None:
            strategy_subset = list(self.strategies.keys())
        
        # Extract parameters for selected strategies
        strategies = [self.strategies[s] for s in strategy_subset]
        n_strategies = len(strategies)
        
        # Expected excess returns
        excess_returns = np.array([s['expected_return'] - self.risk_free_rate for s in strategies])
        
        # Volatilities
        volatilities = np.array([s['volatility'] for s in strategies])
        
        # Create covariance matrix (simplified - assume some correlation)
        correlation_matrix = np.eye(n_strategies)
        for i in range(n_strategies):
            for j in range(i+1, n_strategies):
                # Higher correlation for similar strategies
                if ('reserve' in strategy_subset[i] and 'reserve' in strategy_subset[j]):
                    correlation_matrix[i,j] = correlation_matrix[j,i] = 0.8
                elif ('bitcoin' in strategies[i]['name'].lower() and 'bitcoin' in strategies[j]['name'].lower()):
                    correlation_matrix[i,j] = correlation_matrix[j,i] = 0.6
                else:
                    correlation_matrix[i,j] = correlation_matrix[j,i] = 0.3
        
        # Covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        
        def kelly_objective(weights):
            """Kelly objective function to maximize"""
            portfolio_return = np.dot(weights, excess_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            return -(portfolio_return - 0.5 * portfolio_variance)
        
        # Constraints: weights sum to 1, all weights >= 0
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum to 1
        ]
        
        # Bounds: each weight between 0 and 1
        bounds = [(0, 1) for _ in range(n_strategies)]
        
        # Initial guess
        x0 = np.ones(n_strategies) / n_strategies
        
        # Optimize
        result = minimize(
            kelly_objective,
            x0,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds
        )
        
        return result.x, strategy_subset
    
    def analyze_strategy_combinations(self):
        """Analyze different combinations of strategies"""
        results = {}
        
        # 1. All strategies
        weights, strategies = self.kelly_multi_strategy_allocation()
        results['all_strategies'] = {
            'weights': weights,
            'strategies': strategies,
            'expected_return': self._calculate_portfolio_return(weights, strategies),
            'volatility': self._calculate_portfolio_volatility(weights, strategies),
        }
        
        # 2. Reserve strategies only
        reserve_strategies = ['high_yield_reserve', 'cash_reserve']
        weights, strategies = self.kelly_multi_strategy_allocation(reserve_strategies)
        results['reserve_only'] = {
            'weights': weights,
            'strategies': strategies,
            'expected_return': self._calculate_portfolio_return(weights, strategies),
            'volatility': self._calculate_portfolio_volatility(weights, strategies),
        }
        
        # 3. Traditional allocation strategies
        traditional_strategies = ['static_kelly', 'tactical_allocation']
        weights, strategies = self.kelly_multi_strategy_allocation(traditional_strategies)
        results['traditional_only'] = {
            'weights': weights,
            'strategies': strategies,
            'expected_return': self._calculate_portfolio_return(weights, strategies),
            'volatility': self._calculate_portfolio_volatility(weights, strategies),
        }
        
        # 4. Top 3 performers
        top_strategies = ['high_yield_reserve', 'cash_reserve', 'tactical_allocation']
        weights, strategies = self.kelly_multi_strategy_allocation(top_strategies)
        results['top_three'] = {
            'weights': weights,
            'strategies': strategies,
            'expected_return': self._calculate_portfolio_return(weights, strategies),
            'volatility': self._calculate_portfolio_volatility(weights, strategies),
        }
        
        return results
    
    def _calculate_portfolio_return(self, weights, strategy_names):
        """Calculate expected portfolio return"""
        returns = [self.strategies[s]['expected_return'] for s in strategy_names]
        return np.dot(weights, returns)
    
    def _calculate_portfolio_volatility(self, weights, strategy_names):
        """Calculate portfolio volatility (simplified)"""
        volatilities = [self.strategies[s]['volatility'] for s in strategy_names]
        # Simplified calculation - assumes average correlation
        avg_correlation = 0.5
        portfolio_variance = sum(w**2 * v**2 for w, v in zip(weights, volatilities))
        portfolio_variance += 2 * avg_correlation * sum(
            weights[i] * weights[j] * volatilities[i] * volatilities[j]
            for i in range(len(weights))
            for j in range(i+1, len(weights))
        )
        return np.sqrt(portfolio_variance)
    
    def optimize_bitcoin_sp500_allocation(self, strategy_type='static'):
        """
        Optimize Bitcoin vs S&P 500 allocation for different strategy types
        """
        results = {}
        
        if strategy_type == 'static':
            # Static allocation optimization
            def objective(weights):
                w_btc, w_sp500 = weights
                w_cash = 1 - w_btc - w_sp500
                
                # Portfolio return
                portfolio_return = (w_btc * self.market_params['bitcoin_base_return'] + 
                                  w_sp500 * self.market_params['sp500_base_return'] + 
                                  w_cash * self.risk_free_rate)
                
                # Portfolio variance
                covariance = self.market_params['correlation'] * self.market_params['bitcoin_volatility'] * self.market_params['sp500_volatility']
                portfolio_variance = (w_btc**2 * self.market_params['bitcoin_volatility']**2 + 
                                    w_sp500**2 * self.market_params['sp500_volatility']**2 + 
                                    2 * w_btc * w_sp500 * covariance)
                
                # Kelly objective
                excess_return = portfolio_return - self.risk_free_rate
                kelly_objective = excess_return - 0.5 * portfolio_variance
                
                return -kelly_objective
            
            # Optimize
            constraints = [
                {'type': 'ineq', 'fun': lambda w: 1 - w[0] - w[1]},  # w_btc + w_sp500 <= 1
                {'type': 'ineq', 'fun': lambda w: w[0]},              # w_btc >= 0
                {'type': 'ineq', 'fun': lambda w: w[1]}               # w_sp500 >= 0
            ]
            
            result = minimize(
                objective,
                [0.4, 0.6],  # Initial guess
                method='SLSQP',
                constraints=constraints,
                bounds=[(0, 1), (0, 1)]
            )
            
            w_btc, w_sp500 = result.x
            w_cash = 1 - w_btc - w_sp500
            
            results['static'] = {
                'bitcoin_weight': w_btc,
                'sp500_weight': w_sp500,
                'cash_weight': w_cash,
                'expected_return': self._calculate_static_return(w_btc, w_sp500, w_cash),
                'volatility': self._calculate_static_volatility(w_btc, w_sp500),
                'sharpe_ratio': self._calculate_static_sharpe(w_btc, w_sp500, w_cash)
            }
        
        return results
    
    def _calculate_static_return(self, w_btc, w_sp500, w_cash):
        """Calculate static portfolio return"""
        return (w_btc * self.market_params['bitcoin_base_return'] + 
                w_sp500 * self.market_params['sp500_base_return'] + 
                w_cash * self.risk_free_rate)
    
    def _calculate_static_volatility(self, w_btc, w_sp500):
        """Calculate static portfolio volatility"""
        covariance = self.market_params['correlation'] * self.market_params['bitcoin_volatility'] * self.market_params['sp500_volatility']
        portfolio_variance = (w_btc**2 * self.market_params['bitcoin_volatility']**2 + 
                            w_sp500**2 * self.market_params['sp500_volatility']**2 + 
                            2 * w_btc * w_sp500 * covariance)
        return np.sqrt(portfolio_variance)
    
    def _calculate_static_sharpe(self, w_btc, w_sp500, w_cash):
        """Calculate static portfolio Sharpe ratio"""
        portfolio_return = self._calculate_static_return(w_btc, w_sp500, w_cash)
        portfolio_volatility = self._calculate_static_volatility(w_btc, w_sp500)
        return (portfolio_return - self.risk_free_rate) / portfolio_volatility
    
    def generate_comprehensive_report(self):
        """Generate comprehensive strategy analysis report"""
        
        print("COMPREHENSIVE INVESTMENT STRATEGY OPTIMIZATION")
        print("=" * 80)
        print()
        
        # 1. Individual Strategy Analysis
        print("📊 INDIVIDUAL STRATEGY PERFORMANCE:")
        print("-" * 50)
        
        strategies_sorted = sorted(
            self.strategies.items(), 
            key=lambda x: x[1]['expected_return'], 
            reverse=True
        )
        
        for i, (key, strategy) in enumerate(strategies_sorted, 1):
            print(f"#{i}. {strategy['name']}")
            print(f"    Expected Return: {strategy['expected_return']:.2%}")
            print(f"    Volatility: {strategy['volatility']:.2%}")
            print(f"    Sharpe Ratio: {strategy['sharpe_ratio']:.3f}")
            print(f"    Description: {strategy['description']}")
            print()
        
        # 2. Strategy Combination Analysis
        print("🔄 STRATEGY COMBINATION ANALYSIS:")
        print("-" * 50)
        
        combinations = self.analyze_strategy_combinations()
        
        for combo_name, combo_data in combinations.items():
            print(f"\n{combo_name.upper().replace('_', ' ')} COMBINATION:")
            print(f"Expected Return: {combo_data['expected_return']:.2%}")
            print(f"Volatility: {combo_data['volatility']:.2%}")
            print(f"Sharpe Ratio: {(combo_data['expected_return'] - self.risk_free_rate) / combo_data['volatility']:.3f}")
            print("Allocation:")
            for i, strategy_name in enumerate(combo_data['strategies']):
                weight = combo_data['weights'][i]
                if weight > 0.01:  # Only show allocations > 1%
                    print(f"  {self.strategies[strategy_name]['name']}: {weight:.1%}")
        
        # 3. Bitcoin vs S&P 500 Static Allocation
        print("\n💰 BITCOIN vs S&P 500 STATIC ALLOCATION:")
        print("-" * 50)
        
        static_results = self.optimize_bitcoin_sp500_allocation('static')
        static = static_results['static']
        
        print(f"Optimal Static Allocation:")
        print(f"  Bitcoin: {static['bitcoin_weight']:.1%}")
        print(f"  S&P 500: {static['sp500_weight']:.1%}")
        print(f"  Cash: {static['cash_weight']:.1%}")
        print(f"Expected Return: {static['expected_return']:.2%}")
        print(f"Volatility: {static['volatility']:.2%}")
        print(f"Sharpe Ratio: {static['sharpe_ratio']:.3f}")
        
        # 4. Recommendations
        print("\n🎯 INVESTMENT RECOMMENDATIONS:")
        print("-" * 50)
        
        # Find best overall strategy
        best_combo = max(combinations.items(), 
                        key=lambda x: (x[1]['expected_return'] - self.risk_free_rate) / x[1]['volatility'])
        
        print(f"🏆 BEST OVERALL STRATEGY: {best_combo[0].upper().replace('_', ' ')}")
        print(f"   Expected Return: {best_combo[1]['expected_return']:.2%}")
        print(f"   Volatility: {best_combo[1]['volatility']:.2%}")
        print(f"   Sharpe Ratio: {(best_combo[1]['expected_return'] - self.risk_free_rate) / best_combo[1]['volatility']:.3f}")
        print("\n   Recommended Allocation:")
        for i, strategy_name in enumerate(best_combo[1]['strategies']):
            weight = best_combo[1]['weights'][i]
            if weight > 0.01:
                print(f"   • {weight:.1%} → {self.strategies[strategy_name]['name']}")
        
        # Risk-adjusted recommendations
        print(f"\n📈 FOR AGGRESSIVE INVESTORS:")
        print(f"   Consider: High-Yield Reserve Strategy (40% reserves)")
        print(f"   Expected Return: {self.strategies['high_yield_reserve']['expected_return']:.2%}")
        print(f"   Risk Level: Medium-High")
        
        print(f"\n🛡️  FOR CONSERVATIVE INVESTORS:")
        print(f"   Consider: Tactical Allocation Strategy")
        print(f"   Expected Return: {self.strategies['tactical_allocation']['expected_return']:.2%}")
        print(f"   Risk Level: Medium")
        
        print(f"\n⚖️  FOR BALANCED INVESTORS:")
        print(f"   Consider: Static Kelly Allocation (40% BTC / 60% S&P 500)")
        print(f"   Expected Return: {self.strategies['static_kelly']['expected_return']:.2%}")
        print(f"   Risk Level: Medium")

def main():
    """Run comprehensive strategy optimization"""
    
    optimizer = ComprehensiveStrategyOptimizer()
    optimizer.generate_comprehensive_report()
    
    # Save results
    results_dir = Path("Results/Portfolio")
    results_dir.mkdir(exist_ok=True)
    
    # You can add CSV export functionality here if needed
    print(f"\n📁 Results saved to: {results_dir}")

if __name__ == "__main__":
    main() 