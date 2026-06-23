# Quant Asset Models

A quantitative research framework for modeling the long-term fair value, expected return, and risk
of investments across cryptocurrencies, commodities, and equities. It fits valuation models to
years of price history, simulates forward price paths with Monte Carlo methods, and derives
risk-adjusted portfolio allocations.

> Research and educational project. Nothing here is financial advice.

## What it does

- **Fair-value modeling** — Fits long-term growth models (logarithmic and exponential "rainbow"
  regressions) to each asset's price history to estimate a fair value and quantify how over- or
  under-valued the asset is today.
- **Volatility modeling** — Models volatility as a function of asset maturity (exponential-decay
  fits), capturing how large, established assets tend to become less volatile over time.
- **Forward simulation** — Geometric Brownian Motion (GBM) and stochastic mean-reversion variants
  drive Monte Carlo simulations that produce probabilistic price projections with confidence
  intervals.
- **Portfolio optimization** — Applies the Kelly criterion (single-asset, no-leverage, and
  multi-asset variants) to translate return and volatility estimates into risk-adjusted
  allocations.

## Asset coverage

- **Crypto:** Bitcoin, Ethereum, XRP, Litecoin, Chainlink, Uniswap, AAVE, Quant, RNDR, Ton,
  Injective, Dogecoin, Shiba Inu, Pepe, Floki
- **Commodities:** Gold, Silver, Copper, Platinum, Palladium, Nickel, Lead, Crude Oil, Natural Gas,
  Uranium, Aluminum, and others
- **Equities:** S&P 500

## Methods

| Area        | Techniques                                                                 |
|-------------|----------------------------------------------------------------------------|
| Valuation   | Logarithmic / exponential growth-curve fitting, intercept-slope optimization, overvaluation analysis |
| Risk        | Exponential-decay volatility modeling                                      |
| Simulation  | Geometric Brownian Motion, stochastic mean reversion, Monte Carlo          |
| Allocation  | Kelly criterion (single-asset, no-leverage, three-asset)                   |

## Tech stack

Python · NumPy · pandas · SciPy · scikit-learn · Matplotlib · Seaborn · yfinance

Data sources: Yahoo Finance, Alpha Vantage, FRED, CoinDesk, and metals-price APIs.

## Repository structure

```
Crypto/           Per-asset valuation models and analysis for each cryptocurrency
Commodities/      Per-asset valuation models for each commodity
SP500/            S&P 500 data, models, simulations, and visualizations
Portfolio/        Allocation optimizers (Kelly criterion) and overvaluation tools
model_templates/  Reusable model scaffolds (log, exponential, point-selection)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then add your own API keys
```

All API keys are read from environment variables and are never committed. See `.env.example` for
the required variables.

## Output

For each asset, the framework produces:
- a fitted long-term fair-value curve and an overvaluation estimate versus the current price,
- Monte Carlo price projections with confidence intervals, and
- risk-adjusted portfolio allocations (expected return, volatility, Sharpe ratio) derived from those inputs.

All figures are model output for research purposes, not predictions.
