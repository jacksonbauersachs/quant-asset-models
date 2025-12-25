import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def analyze_current_market_conditions():
    """Analyze current market conditions based on known data"""
    print("CURRENT MARKET ANALYSIS")
    print("="*60)
    
    # Current market data (as of July 2024)
    current_price = 118000  # Current Bitcoin price
    current_volatility = 45  # Current annualized volatility (much lower than historical)
    recent_growth_rate = 25  # Recent annualized growth rate
    
    # Market condition indicators
    print(f"\nMarket Conditions:")
    print(f"- Price: ${current_price:,.2f}")
    print(f"- Volatility: {current_volatility:.1f}% (much lower than historical 80-120%)")
    print(f"- Recent Growth: {recent_growth_rate:.1f}% annualized")
    
    # Volatility comparison
    print(f"\nVolatility Context:")
    print(f"- Current: {current_volatility:.1f}%")
    print(f"- Historical (2017-2021): 80-120%")
    print(f"- Market has matured significantly")
    
    # Price context
    fair_value_estimate = 107000  # Based on your previous analysis
    overvaluation = ((current_price / fair_value_estimate) - 1) * 100
    
    print(f"\nValuation Context:")
    print(f"- Current price: ${current_price:,.2f}")
    print(f"- Estimated fair value: ${fair_value_estimate:,.2f}")
    print(f"- Overvaluation: {overvaluation:.1f}%")
    
    return current_price, current_volatility, recent_growth_rate, overvaluation

def create_realistic_scenarios(current_price, volatility, growth_rate):
    """Create realistic market scenarios based on current conditions"""
    print("\n" + "="*60)
    print("REALISTIC MARKET SCENARIOS")
    print("="*60)
    
    scenarios = {}
    
    # Scenario 1: Continuation (current trend continues)
    scenarios['Continuation'] = {
        'growth_rate': growth_rate * 0.8,  # Slightly slower than recent
        'volatility': volatility,
        'description': 'Current trend continues, moderate growth'
    }
    
    # Scenario 2: Correction then Recovery
    scenarios['Correction_Recovery'] = {
        'growth_rate': growth_rate * 0.5,  # Slower growth after correction
        'volatility': volatility * 1.5,    # Higher volatility during correction
        'description': '20-30% correction, then recovery'
    }
    
    # Scenario 3: Sideways Market
    scenarios['Sideways'] = {
        'growth_rate': 0,  # No growth
        'volatility': volatility * 0.8,  # Lower volatility
        'description': 'Trading in a range for 1-2 years'
    }
    
    # Scenario 4: Accelerated Growth
    scenarios['Accelerated'] = {
        'growth_rate': growth_rate * 1.2,  # Faster growth
        'volatility': volatility * 1.3,    # Higher volatility
        'description': 'Strong institutional adoption drives growth'
    }
    
    print("Scenario Probabilities (based on current market conditions):")
    print("- Continuation: 40% (most likely)")
    print("- Correction then Recovery: 30%")
    print("- Sideways: 20%")
    print("- Accelerated: 10%")
    
    return scenarios

def simulate_lump_sum_timing_strategies(current_price, scenarios, lump_sum_amount=10000):
    """Simulate different lump sum timing strategies across scenarios"""
    print("\n" + "="*60)
    print("LUMP SUM TIMING STRATEGY ANALYSIS")
    print("(Including 5% interest on waiting funds)")
    print("="*60)
    
    strategies = {
        'Invest All Immediately': {'immediate': 1.0, 'wait_drop': None},
        'Wait for 5% drop': {'immediate': 0.0, 'wait_drop': 0.05},
        'Wait for 10% drop': {'immediate': 0.0, 'wait_drop': 0.10},
        'Wait for 15% drop': {'immediate': 0.0, 'wait_drop': 0.15},
        'Wait for 20% drop': {'immediate': 0.0, 'wait_drop': 0.20},
        'Wait for 25% drop': {'immediate': 0.0, 'wait_drop': 0.25},
        '50% Immediate + 50% Wait 10%': {'immediate': 0.5, 'wait_drop': 0.10},
        '70% Immediate + 30% Wait 15%': {'immediate': 0.7, 'wait_drop': 0.15},
        '30% Immediate + 70% Wait 20%': {'immediate': 0.3, 'wait_drop': 0.20}
    }
    
    results = {}
    
    for scenario_name, scenario_params in scenarios.items():
        print(f"\n--- {scenario_name} Scenario ---")
        print(f"Growth rate: {scenario_params['growth_rate']:.1f}%")
        print(f"Volatility: {scenario_params['volatility']:.1f}%")
        
        # Simulate 1000 paths for this scenario
        n_paths = 1000
        n_days = 365 * 2  # 2 years
        
        # Generate price paths
        daily_return = scenario_params['growth_rate'] / 252 / 100
        daily_vol = scenario_params['volatility'] / np.sqrt(252) / 100
        
        price_paths = np.zeros((n_days, n_paths))
        price_paths[0, :] = current_price
        
        for day in range(1, n_days):
            returns = np.random.normal(daily_return, daily_vol, n_paths)
            price_paths[day, :] = price_paths[day-1, :] * (1 + returns)
        
        # Test each strategy
        scenario_results = {}
        
        for strategy_name, strategy_params in strategies.items():
            final_values = np.zeros(n_paths)
            
            for path in range(n_paths):
                path_prices = price_paths[:, path]
                total_coins = 0
                total_cash = 0
                
                # Immediate investment portion
                if strategy_params['immediate'] > 0:
                    immediate_amount = lump_sum_amount * strategy_params['immediate']
                    coins_bought = immediate_amount / path_prices[0]
                    total_coins += coins_bought
                
                # Wait for drop portion
                if strategy_params['wait_drop'] is not None:
                    wait_amount = lump_sum_amount * (1 - strategy_params['immediate'])
                    target_price = current_price * (1 - strategy_params['wait_drop'])
                    invested = False
                    days_waited = 0
                    
                    for day, price in enumerate(path_prices):
                        if price <= target_price and not invested:
                            # Calculate interest earned while waiting
                            interest_earned = wait_amount * (0.05 / 365) * days_waited
                            total_investment = wait_amount + interest_earned
                            
                            coins_bought = total_investment / price
                            total_coins += coins_bought
                            invested = True
                            break
                        days_waited += 1
                    
                    if not invested:  # Never hit target, invest at end with accumulated interest
                        interest_earned = wait_amount * (0.05 / 365) * len(path_prices)
                        total_investment = wait_amount + interest_earned
                        coins_bought = total_investment / path_prices[-1]
                        total_coins += coins_bought
                
                final_values[path] = total_coins * path_prices[-1]
            
            # Calculate statistics
            returns = (final_values / lump_sum_amount - 1) * 100
            scenario_results[strategy_name] = {
                'mean_return': np.mean(returns),
                'median_return': np.median(returns),
                'std_return': np.std(returns),
                'best_case': np.max(returns),
                'worst_case': np.min(returns),
                'final_value_mean': np.mean(final_values)
            }
        
        results[scenario_name] = scenario_results
    
    return results

def analyze_results(results, scenarios):
    """Analyze and present the results"""
    print("\n" + "="*80)
    print("COMPREHENSIVE LUMP SUM TIMING STRATEGY RESULTS")
    print("="*80)
    
    # Scenario probabilities
    scenario_probs = {
        'Continuation': 0.40,
        'Correction_Recovery': 0.30,
        'Sideways': 0.20,
        'Accelerated': 0.10
    }
    
    # Calculate weighted average returns for each strategy
    strategy_summary = {}
    
    for strategy_name in results['Continuation'].keys():
        weighted_return = 0
        weighted_std = 0
        
        for scenario_name, prob in scenario_probs.items():
            weighted_return += results[scenario_name][strategy_name]['mean_return'] * prob
            weighted_std += results[scenario_name][strategy_name]['std_return'] * prob
        
        strategy_summary[strategy_name] = {
            'weighted_return': weighted_return,
            'weighted_std': weighted_std,
            'risk_adjusted_return': weighted_return / weighted_std if weighted_std > 0 else 0
        }
    
    # Sort by weighted return
    sorted_strategies = sorted(strategy_summary.items(), 
                             key=lambda x: x[1]['weighted_return'], reverse=True)
    
    print(f"\n{'Strategy':<35} {'Weighted Return':<15} {'Risk-Adjusted':<15} {'Risk Level':<10}")
    print("-" * 80)
    
    for strategy_name, stats in sorted_strategies:
        risk_level = "Low" if stats['weighted_std'] < 30 else "Medium" if stats['weighted_std'] < 50 else "High"
        print(f"{strategy_name:<35} {stats['weighted_return']:>13.1f}% {stats['risk_adjusted_return']:>13.2f} {risk_level:>10}")
    
    # Detailed scenario breakdown
    print(f"\n" + "="*80)
    print("DETAILED SCENARIO BREAKDOWN")
    print("="*80)
    
    for scenario_name, prob in scenario_probs.items():
        print(f"\n{scenario_name} Scenario (Probability: {prob*100:.0f}%)")
        print("-" * 60)
        
        # Sort strategies by return in this scenario
        scenario_results = results[scenario_name]
        sorted_in_scenario = sorted(scenario_results.items(), 
                                  key=lambda x: x[1]['mean_return'], reverse=True)
        
        for i, (strategy_name, stats) in enumerate(sorted_in_scenario[:5]):  # Top 5
            print(f"{i+1}. {strategy_name:<30} {stats['mean_return']:>8.1f}% (Best: {stats['best_case']:>6.1f}%, Worst: {stats['worst_case']:>6.1f}%)")
    
    # Recommendations
    print(f"\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    best_strategy = sorted_strategies[0]
    print(f"\nBest Overall Strategy: {best_strategy[0]}")
    print(f"Weighted Return: {best_strategy[1]['weighted_return']:.1f}%")
    print(f"Risk Level: {'Low' if best_strategy[1]['weighted_std'] < 30 else 'Medium' if best_strategy[1]['weighted_std'] < 50 else 'High'}")
    
    # Conservative recommendation
    conservative_strategies = [s for s in sorted_strategies if s[1]['weighted_std'] < 40]
    if conservative_strategies:
        conservative_best = conservative_strategies[0]
        print(f"\nConservative Choice: {conservative_best[0]}")
        print(f"Weighted Return: {conservative_best[1]['weighted_return']:.1f}%")
        print(f"Risk Level: Low")
    
    # Key insights
    print(f"\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    # Find immediate vs waiting strategies
    immediate_strategies = [s for s in sorted_strategies if 'Immediately' in s[0] and 'Wait' not in s[0]]
    waiting_strategies = [s for s in sorted_strategies if 'Wait for' in s[0] and 'Immediate' not in s[0]]
    hybrid_strategies = [s for s in sorted_strategies if 'Immediate' in s[0] and 'Wait' in s[0]]
    
    if immediate_strategies:
        best_immediate = immediate_strategies[0]
        print(f"\nBest Immediate Strategy: {best_immediate[0]}")
        print(f"Return: {best_immediate[1]['weighted_return']:.1f}%")
    
    if waiting_strategies:
        best_waiting = waiting_strategies[0]
        print(f"\nBest Waiting Strategy: {best_waiting[0]}")
        print(f"Return: {best_waiting[1]['weighted_return']:.1f}%")
    
    if hybrid_strategies:
        best_hybrid = hybrid_strategies[0]
        print(f"\nBest Hybrid Strategy: {best_hybrid[0]}")
        print(f"Return: {best_hybrid[1]['weighted_return']:.1f}%")
    
    return strategy_summary

def main():
    """Main analysis function"""
    print("BITCOIN LUMP SUM TIMING ANALYSIS")
    print("Immediate vs Waiting for Drops (5% interest on waiting funds)")
    print("="*60)
    
    # Get current market data
    current_price, volatility, growth_rate, overvaluation = analyze_current_market_conditions()
    
    # Create realistic scenarios
    scenarios = create_realistic_scenarios(current_price, volatility, growth_rate)
    
    # Simulate strategies
    results = simulate_lump_sum_timing_strategies(current_price, scenarios)
    
    # Analyze results
    strategy_summary = analyze_results(results, scenarios)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'Results/Bitcoin/lump_sum_timing_analysis_{timestamp}.txt'
    
    with open(output_file, 'w') as f:
        f.write("BITCOIN LUMP SUM TIMING ANALYSIS\n")
        f.write("Immediate vs Waiting for Drops (5% interest on waiting funds)\n")
        f.write("="*60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Current Bitcoin Price: ${current_price:,.2f}\n")
        f.write(f"Current Volatility: {volatility:.1f}%\n")
        f.write(f"Recent Growth Rate: {growth_rate:.1f}%\n")
        f.write(f"Overvaluation: {overvaluation:.1f}%\n")
        f.write("="*60 + "\n\n")
        
        # Write strategy summary
        f.write("STRATEGY RANKINGS (Weighted by Scenario Probability)\n")
        f.write("-"*60 + "\n")
        sorted_strategies = sorted(strategy_summary.items(), 
                                 key=lambda x: x[1]['weighted_return'], reverse=True)
        
        for strategy_name, stats in sorted_strategies:
            risk_level = "Low" if stats['weighted_std'] < 30 else "Medium" if stats['weighted_std'] < 50 else "High"
            f.write(f"{strategy_name:<35} {stats['weighted_return']:>13.1f}% {stats['risk_adjusted_return']:>13.2f} {risk_level:>10}\n")
    
    print(f"\nDetailed results saved to: {output_file}")
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 