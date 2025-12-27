# Bitcoin Regime Shift Analysis - Executive Summary

**Analysis Date:** December 26, 2025  
**Data Range:** October 1, 2017 to December 24, 2025  
**Total Data Points:** 3,007 daily prices

---

## Overview

This analysis implements two complementary approaches to address the regime shift problem in Bitcoin's growth model:

1. **Rolling Regression Analysis** - Tracks how model parameters change over time to identify regime shifts
2. **Trend + Cycle Model** - Fits a modern trend and creates valuation bands based on cyclical deviations

---

## Key Findings

### 1. Rolling Regression Analysis

**3-Year Rolling Window:**
- Analyzed 64 windows
- Average slope: 1.78 (range: 0.10 to 4.35)
- Average R²: -0.12 (range: -3.21 to 0.93)
- **Recent trend (last 2 years):** Slope = 1.10, R² = -0.32

**5-Year Rolling Window:**
- Analyzed 40 windows  
- Average slope: 1.91 (range: 0.58 to 2.89)
- Average R²: 0.36 (range: 0.03 to 0.59)
- **Recent trend (last 2 years):** Slope = 2.24, R² = 0.29

**Key Insight:** The 3-year window shows **negative R² in recent periods**, indicating the single-slope model no longer fits well. This is strong evidence of a regime shift.

### 2. Identified Regime Shifts

**27 significant regime shifts detected** (>20% slope change):

**Major Shifts:**
- **2019-2020:** Multiple volatility spikes during early adoption phase
- **2020-2021:** Institutional wave (slope increased to 3.4+)
- **2022:** Bear market crash (slope dropped to 0.10)
- **2023-2024:** Recovery and ETF era (slope increased to 4.3+)

**Most Recent Shifts:**
- **2024-01-05:** Slope jumped from 1.69 to 2.89 (+70.7%) - ETF approval impact
- **2024-03-05:** Slope increased to 3.49 (+20.5%)
- **2024-06-03:** Slope peaked at 4.35 (+20.6%)

### 3. Trend + Cycle Model (Modern Regime)

**Model Parameters:**
- **Slope:** 1.70
- **Intercept:** -9.74
- **R²:** 0.76 (fits modern regime well)
- **Trend Period:** October 1, 2017 to December 24, 2025

**Current Valuation (Dec 24, 2025):**
- **Current Price:** $87,611.96
- **Model Fair Value:** $86,727.11
- **Deviation:** +1.02%
- **Z-Score:** 0.04 standard deviations
- **Status:** **Hold (Fair Value)**

**Valuation Bands:**
- **-2 std dev (Firesale!):** $32,434.95
- **-1 std dev (Buy):** $52,770.70
- **Trend (Fair):** $86,727.11
- **+1 std dev (Sell):** $139,685.73
- **+2 std dev (Danger!):** $227,264.51

---

## Interpretation

### What This Means

1. **Regime Shifts Are Real:** The rolling regression shows clear evidence that Bitcoin's growth dynamics have fundamentally changed multiple times. The single-slope model is **not appropriate** for the entire history.

2. **Recent Model Breakdown:** The negative R² in recent 3-year windows indicates the model is breaking down. This suggests we're in a new regime that doesn't follow the old pattern.

3. **Modern Trend is More Defensible:** The trend + cycle model uses only post-2017 data (modern regime) and achieves R² = 0.76, which is much better than trying to fit all historical data.

4. **Current Valuation:** Bitcoin is currently at **fair value** relative to the modern trend. It's not significantly overvalued or undervalued.

### Why This Matters

- **Avoids Anchoring Bias:** Doesn't anchor to early low-liquidity years (2010-2014)
- **Acknowledges Maturity:** Recognizes that Bitcoin has matured and growth has slowed
- **Actionable Signals:** Provides clear "cheap/fair/rich" bands based on recent volatility
- **Regime-Aware:** Uses only data from the current regime, not historical averages

---

## Recommendations

### For Model Selection

1. **Use Trend + Cycle Model** for current valuation decisions
   - More defensible than single-slope model
   - Acknowledges regime shifts
   - Provides actionable bands

2. **Avoid Single-Slope Model** for all-time data
   - R² is negative in recent periods
   - Anchors to early low-liquidity years
   - Doesn't account for maturity

3. **Consider Piecewise Models** if you need historical context
   - Use identified breakpoints: 2017-10-01, 2020-03-01, 2024-01-01
   - Fit separate segments for each regime
   - Compare out-of-sample performance

### For Current Position

- **Current Status:** Hold (Fair Value)
- **Action:** Wait for clearer signals
- **Buy Signal:** If price drops to ~$53K (-1 std dev)
- **Sell Signal:** If price rises to ~$140K (+1 std dev)

---

## Technical Details

### Methodology

**Rolling Regression:**
- 3-year and 5-year rolling windows
- Slides forward by 30 days
- Fits logarithmic model: `price = 10^(slope × ln(days) + intercept)`
- Identifies shifts where slope changes >20%

**Trend + Cycle Model:**
- Fits trend to modern regime (post-2017)
- Calculates residuals in log space
- Creates bands at ±1σ and ±2σ
- Determines valuation status based on z-score

### Files Generated

1. `bitcoin_regime_analysis.png` - Visualizations of slope/R² over time
2. `bitcoin_regime_findings.txt` - Detailed rolling regression report
3. `bitcoin_trend_cycle_model.png` - Trend + cycle visualization with bands
4. `bitcoin_valuation_report.txt` - Current valuation analysis

---

## Conclusion

The analysis confirms that **Bitcoin has experienced multiple regime shifts** and that a single-slope model for all-time data is not appropriate. The **Trend + Cycle Model** provides a more defensible approach for current valuation decisions by:

1. Using only modern regime data (post-2017)
2. Acknowledging that growth dynamics have changed
3. Providing actionable valuation bands
4. Avoiding anchoring to early low-liquidity years

**Current Bitcoin valuation: Fair Value** - No strong buy or sell signal at this time.

---

*This analysis was generated using Python scripts that implement rolling regression and trend+cycle modeling. All code is available in the `log_model/` folder.*

