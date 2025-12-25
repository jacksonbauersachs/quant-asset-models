# Bitcoin Analysis Workflow - Procedural Scripts
**Purpose: Organized scripts for replicating the complete Bitcoin analysis workflow**

## Quick Start
Run these scripts in order to replicate the complete Bitcoin analysis:

1. `01_fetch_bitcoin_data.py` - Get latest Bitcoin price data
2. `02_clean_combine_data.py` - Clean and combine historical data
3. `03_fit_growth_model.py` - Develop logarithmic growth model
4. `04_fit_volatility_model.py` - Develop exponential decay volatility model
5. `05_run_monte_carlo.py` - Execute Monte Carlo simulation
6. `06_verify_results.py` - Verify and test all models

## What Each Script Does

### Data Collection & Cleaning
- **01_fetch_bitcoin_data.py**: Fetches latest Bitcoin price data from API
- **02_clean_combine_data.py**: Cleans, combines, and prepares historical data

### Model Development
- **03_fit_growth_model.py**: Fits logarithmic growth model (log10(price) = a*ln(day) + b)
- **04_fit_volatility_model.py**: Fits exponential decay volatility model (vol = a*exp(-b*years) + c)

### Simulation & Analysis
- **05_run_monte_carlo.py**: Runs Monte Carlo simulation with growth and volatility models
- **06_verify_results.py**: Tests and verifies all models and calculations

## Key Files Generated
- Growth model coefficients: `Models/Growth Models/bitcoin_growth_model_coefficients.txt`
- Volatility model results: `Models/Volatility Models/bitcoin_exponential_volatility_results_YYYYMMDD.txt`
- Simulation results: `Results/Bitcoin/bitcoin_monte_carlo_simple_*.csv`
- Visualizations: `Results/Bitcoin/bitcoin_monte_carlo_simple_visualization_YYYYMMDD.png`

## Important Notes
- All scripts include detailed logging and error handling
- Day numbering is critical: use days since Bitcoin genesis (Jan 3, 2009)
- Current day number: 6041 (as of 2025)
- Volatility calculated from Bitcoin's "start" (July 18, 2010)
- Current Bitcoin age: ~15 years

## Dependencies
- pandas, numpy, matplotlib, scipy
- datetime for date handling
- API access for Bitcoin data

## Troubleshooting
- Check API rate limits when fetching data
- Verify day numbering calculations
- Ensure volatility caps at 100% (not 200%)
- Monitor for unrealistic price predictions

## Documentation
See `Documentation/bitcoin_analysis_workflow_2025.md` for complete workflow documentation. 