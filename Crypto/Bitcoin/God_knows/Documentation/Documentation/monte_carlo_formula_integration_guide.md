# Bitcoin Monte Carlo Simulation & Formula Integration Guide

## Overview
This document explains how the Bitcoin Monte Carlo simulation integrates with the growth and volatility decay formulas, including key insights learned during development and debugging.

## Key Formulas

### 1. Growth Formula
```
log10(price) = 1.827743 * ln(day) + (-10.880943)
```
- **Purpose**: Predicts Bitcoin's expected price at any given day
- **Day Numbering**: Uses days since Bitcoin genesis (January 3, 2009)
- **R²**: 0.9403 (excellent fit)
- **Range**: day >= 365 (starts from Bitcoin's second year)

### 2. Volatility Decay Formula
```
volatility = 2.310879 * exp(-0.124138 * years) + 0.077392
```
- **Purpose**: Predicts Bitcoin's volatility at any future time point
- **Type**: Exponential decay model
- **R²**: 0.2883
- **Behavior**: Starts high (~239%) and decays to ~7.7% long-term
- **Half-life**: 5.58 years

## Critical Day Numbering Issue (RESOLVED)

### The Problem
Initially, we incorrectly calculated the day number for the growth formula, leading to:
- **Wrong prediction**: $53,263 (121.7% difference from actual)
- **Incorrect overvaluation**: Believed Bitcoin was massively overvalued

### The Solution
**Correct day numbering**: Use days since Bitcoin genesis (January 3, 2009)
- **Today's formula day**: 6041 (as of July 19, 2025)
- **Correct prediction**: $107,641 (only 9.7% difference from actual $118,075)
- **Actual overvaluation**: Bitcoin is only slightly overvalued relative to formula

### Why This Matters
The day numbering directly affects:
1. Current price predictions
2. Future growth rate calculations
3. Monte Carlo simulation accuracy
4. Investment timing decisions

## Monte Carlo Simulation Architecture

### Approach: Simple Target-Based Simulation
Instead of complex GBM with dynamic growth rates, we use:
1. **Formula predictions as targets**: Calculate expected price at each time point
2. **Volatility-driven randomness**: Add random variations around the target
3. **Natural convergence**: Current price gradually moves toward formula predictions

### Key Components

#### 1. Growth Rate Calculation
```python
# Calculate expected price at future time
future_day = today_day + int(years * 365.25)
expected_price = 10**(a * np.log(future_day) + b)

# Use formula predictions as targets, not growth rates
```

#### 2. Volatility Integration
```python
# Get volatility at specific time point
volatility = a_vol * np.exp(-b_vol * years) + c_vol

# Add random volatility around expected price
random_factor = np.exp(np.random.normal(0, volatility * sqrt(dt)))
```

#### 3. Price Evolution
```python
# Gradual convergence toward formula predictions
weight = min(0.1, dt)  # Small weight for gradual movement
target_price = (1 - weight) * current_price + weight * expected_price
new_price = target_price * random_factor
```

## Economic Logic

### Why This Approach Makes Sense

1. **Formula as Baseline**: The growth formula represents Bitcoin's fundamental value trajectory
2. **Current Overvaluation**: Bitcoin trades above formula prediction (9.7% overvalued)
3. **Natural Convergence**: Market forces should pull price toward fundamental value
4. **Volatility Decay**: Decreasing volatility reflects Bitcoin's maturation
5. **Short-term vs Long-term**: Allows for short-term deviations while maintaining long-term trend

### Expected Behavior
- **Short-term**: Price may decline toward formula baseline
- **Medium-term**: Price follows formula's growth trajectory
- **Long-term**: Price converges to formula predictions with decreasing volatility

## Implementation Details

### Data Sources
- **Bitcoin Data**: `Bitcoin_Final_Complete_Data_20250719.csv` (5,481 days)
- **Growth Model**: `bitcoin_growth_model_coefficients_day365.txt`
- **Volatility Model**: `bitcoin_exponential_volatility_results_20250719.txt`

### Key Parameters
- **Starting Price**: $118,075 (actual current price)
- **Simulation Period**: 10 years
- **Number of Paths**: 1,000
- **Time Step**: Daily (1/365.25 years)

### Output Analysis
- **Mean/Median**: Central tendency of price paths
- **Percentiles (5th, 95th)**: Confidence intervals
- **Standard Deviation**: Measure of uncertainty
- **Visualization**: Price paths, distributions, volatility decay

## Lessons Learned

### 1. Day Numbering is Critical
- Always verify day numbering against actual dates
- Growth formula uses days since Bitcoin genesis
- Incorrect day numbering leads to massive prediction errors

### 2. Formula Integration Strategy
- Use formulas as targets, not just growth rates
- Allow natural convergence toward fundamental value
- Combine growth predictions with volatility-driven randomness

### 3. Economic Realism
- Bitcoin's current overvaluation is modest (9.7%), not extreme (121.7%)
- Volatility decay reflects asset maturation
- Short-term deviations are expected and realistic

### 4. Simulation Design
- Simple approaches often work better than complex GBM
- Target-based simulation captures economic intuition
- Gradual convergence prevents unrealistic price jumps

## Future Improvements

### Potential Enhancements
1. **Regime Detection**: Different volatility regimes (bull/bear markets)
2. **External Factors**: Integration with macro indicators
3. **Confidence Intervals**: Dynamic confidence bands based on volatility
4. **Scenario Analysis**: Different growth rate scenarios

### Validation Methods
1. **Backtesting**: Test against historical data
2. **Out-of-Sample Testing**: Validate on unseen data
3. **Sensitivity Analysis**: Test parameter robustness
4. **Economic Validation**: Ensure predictions align with economic theory

## Usage Guidelines

### When to Use This Simulation
- **Long-term planning**: 5-10 year investment horizons
- **Risk assessment**: Understanding price uncertainty
- **Timing decisions**: Identifying potential entry/exit points
- **Portfolio allocation**: Bitcoin allocation decisions

### Limitations
- **Not for short-term trading**: Designed for long-term analysis
- **Assumes formula validity**: Depends on growth formula accuracy
- **Simplified volatility**: Uses single decay model
- **No external shocks**: Doesn't account for black swan events

## Conclusion

The integrated Monte Carlo simulation successfully combines:
- **Growth formula predictions** as fundamental value targets
- **Volatility decay model** for realistic uncertainty
- **Natural convergence** toward formula predictions
- **Economic realism** in price behavior

This approach provides a robust framework for Bitcoin price forecasting that respects both mathematical rigor and economic intuition. 