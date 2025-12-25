# Mean Reversion Theory Analysis

This folder contains all the theoretical analysis and modeling related to Bitcoin's mean reversion behavior.

## 📁 **Scripts**

### **Core Analysis Scripts:**
- `analyze_mean_reversion_daily.py` - Fine-resolution daily mean reversion analysis
- `analyze_mean_reversion_rate_of_change.py` - Analysis of how mean reversion speed changes over time
- `bitcoin_dual_decay_mean_reversion.py` - Dual decay model for mean reversion speed and volatility
- `bitcoin_mean_reversion_backtest.py` - Statistical backtest of mean reversion strategy

### **GBM Simulation Scripts:**
- `bitcoin_stochastic_mean_reversion_gbm.py` - Stochastic mean reversion GBM model
- `bitcoin_realistic_mean_reversion_gbm.py` - Realistic mean reversion starting at market price
- `bitcoin_statistically_validated_gbm.py` - GBM using statistically validated mean reversion factors

## 📊 **Key Findings**

### **Statistically Validated Mean Reversion Factors:**

| Price/Fair Value Ratio | Mean Return (90 days) | Statistical Significance | Mean Reversion Speed (λ) |
|------------------------|----------------------|-------------------------|--------------------------|
| **>1.5x (Very Overvalued)** | **-35.1%** | **p < 0.001** ✅ | **90** |
| **1.2-1.5x (Moderately Overvalued)** | **-4.9%** | **p = 0.027** ✅ | **60** |
| **1.0-1.2x (Slightly Overvalued)** | **+0.6%** | **p = 0.785** ❌ | **15** |
| **0.8-1.0x (Fair Value)** | **+3.8%** | **p = 0.785** ❌ | **9** |
| **0.5-0.8x (Moderately Undervalued)** | **+25.6%** | **p < 0.001** ✅ | **75** |
| **<0.5x (Very Undervalued)** | **+99.4%** | **p < 0.001** ✅ | **120** |

### **Mean Reversion Speed Evolution:**
- **Early Bitcoin**: High mean reversion speed (λ ≈ 50-100)
- **Mature Bitcoin**: Lower mean reversion speed (λ ≈ 20-40)
- **Future Bitcoin**: Expected to be more stable (λ ≈ 10-30)

## 📈 **Data Files**

### **Analysis Results:**
- `bitcoin_mean_reversion_daily_*.csv` - Daily mean reversion calculations
- `bitcoin_mean_reversion_weekly_*.csv` - Weekly mean reversion calculations
- `bitcoin_mean_reversion_rates_*.csv` - Rate of change analysis
- `bitcoin_mean_reversion_backtest_*.csv` - Backtest results

### **Visualizations:**
- `bitcoin_mean_reversion_fine_resolution_*.png` - Fine-resolution analysis
- `bitcoin_mean_reversion_statistical_*.png` - Statistical comparison
- `bitcoin_dual_decay_analysis_*.png` - Dual decay model visualization
- `bitcoin_mean_reversion_backtest_*.png` - Backtest results visualization

## 🎯 **Investment Implications**

### **Strong Signals (Statistically Significant):**
- **Very Overvalued (>1.5x)**: Strong sell signal
- **Very Undervalued (<0.5x)**: Strong buy signal
- **Moderately Undervalued (0.5-0.8x)**: Strong buy signal

### **Weak Signals (Not Statistically Significant):**
- **Slightly Overvalued (1.0-1.2x)**: No clear signal
- **Fair Value (0.8-1.0x)**: No clear signal

### **Current Bitcoin (1.1x overvalued):**
- **Historical evidence**: No statistically significant pattern
- **Trend**: Slightly negative (median -8% return)
- **Recommendation**: HOLD, monitor for 1.2x threshold

## 🔬 **Methodology**

### **Mean Reversion Calculation:**
1. Calculate fair value using growth formula
2. Compute price/fair value ratio
3. Calculate autocorrelation of log price ratios
4. Derive mean reversion speed: λ = -ln(autocorrelation)

### **Statistical Validation:**
1. Backtest mean reversion strategy over 5,361 observations
2. Use t-tests to determine statistical significance
3. Categorize by price/fair value ratio thresholds
4. Calculate expected returns for each category

### **GBM Integration:**
1. Dynamic mean reversion speed based on price ratio
2. Stochastic evolution of mean reversion parameters
3. Integration with growth and volatility models
4. Monte Carlo simulation with 1,000+ paths

## 📚 **Academic Context**

This analysis builds on:
- **Ornstein-Uhlenbeck Process**: Mean reverting stochastic processes
- **Autocorrelation Analysis**: Standard method for detecting mean reversion
- **Statistical Backtesting**: Validation against historical data
- **Dynamic Parameter Models**: Time-varying mean reversion speed

## 🚨 **Limitations**

1. **Regime Changes**: Bitcoin's behavior may change as it matures
2. **Sample Size**: Limited historical data for extreme cases
3. **Model Complexity**: Multiple stochastic processes increase uncertainty
4. **Market Evolution**: Institutional adoption may alter mean reversion patterns

## 📝 **Usage**

To run any analysis:
```bash
python script_name.py
```

Results will be saved to the same folder with timestamps. 