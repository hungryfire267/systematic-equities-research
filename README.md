# ASX Alpha System: A Systematic Long-Short Equities Research Pipeline

**An end-to-end quantitative research pipeline covering ~200 ASX-listed equities — from raw price/volume data to a statistically validated, long-short portfolio, deployed as an interactive dashboard.**

Built solo, from data ingestion through model deployment, to practice the full quant research workflow: signal construction, rigorous out-of-sample validation, multiple-testing correction, and portfolio optimization.

> **TL;DR for recruiters:** 40+ engineered factors, purged walk-forward validation with embargo, Benjamini–Hochberg FDR correction across all signal tests, three tree-based model architectures compared via paired statistical tests (Newey-West HAC, McNemar), and a mean-variance long-short portfolio construction layer. Out-of-sample Sharpe ratio of **1.50**. Full pipeline is scripted and reproducible — no notebook-only analysis.

---

## Why this project

Most portfolio-style ML repos stop at "trained a model, backtested it, got a nice equity curve." The equity curve alone doesn't tell you whether a signal is real or a multiple-testing artifact. This project was built to close that gap — every signal is tested for statistical significance *before* it's allowed into the feature set, and every model comparison is backed by a formal hypothesis test rather than eyeballing which line is higher on a chart.

---

## Architecture

The pipeline is organised as a linear, script-driven flow (not a notebook chain), so each stage can be run, tested, and re-run independently:

```
Raw data ingestion (yfinance, RBA/ABS macro series)
        │
        ▼
Signal construction  (13 signal families → 40+ parameterised factors)
        │
        ▼
Information Coefficient testing  (Spearman rank-IC, Newey-West HAC t-tests,
                                   Benjamini-Hochberg FDR correction)
        │
        ▼
Feature matrix construction  (stock-level / market-level / macro+market-level)
        │
        ▼
Model training  (Decision Tree · LightGBM · XGBoost · GRU · Linear baseline)
        │  purged, embargoed walk-forward validation
        ▼
Model comparison  (paired t-tests on IC, McNemar test on hit rate)
        │
        ▼
Portfolio construction  (top/bottom-20 rank selection → mean-variance optimiser)
        │
        ▼
Streamlit dashboard  (Docker-containerised, deployed on AWS)
```

### Repository structure

```
scripts/
├── signals/            # 13 signal families: momentum, reversal, microstructure,
│                        #   beta, autocorrelation, PVO, trend, mean-volatility,
│                        #   relative strength, pairs trading, GARCH-based market
│                        #   vol (arch), momentum×liquidity & reversal×illiquidity
│                        #   interaction factors
├── macro/               # RBA/ABS macro series ingestion (interest rates, credit,
│                        #   VIX)
├── preprocessing/        # Feature matrix assembly, forward-return target
│                        #   construction, cross-sectional signal cleaning
├── research/             # IC calculation, Newey-West HAC t-tests, FDR correction,
│                        #   quintile spread tests, diagnostic plotting
├── models/                # DT / LightGBM / XGBoost / GRU / Linear models +
│                        #   Optuna-style random-search tuners + purged
│                        #   walk-forward validator
├── portfolio/             # Top/bottom-N selection, mean-variance optimiser
│                        #   (SciPy SLSQP), backtest metrics, hypothesis tests
│                        #   comparing models
└── dashboard/            # Metric aggregation feeding the Streamlit app

app/                      # Multi-page Streamlit dashboard (Portfolio, Backtest
                           #   Performance, Model Comparison, Methodology)
notebooks/                # Exploratory research (signal design, model
                           #   comparison drafts)
tests/                    # Pipeline unit tests
```

---

## Methodology

### 1. Signal construction (40+ factors, 13 families)
Cross-sectionally ranked, parameterised factors spanning:
- **Momentum & reversal** across multiple lookback windows, plus liquidity- and illiquidity-interacted variants (momentum×volume, reversal×Amihud illiquidity)
- **Microstructure** — volume-based and price-impact signals
- **Volatility & risk** — mean volatility, GARCH-modelled market volatility (via `arch`), beta exposure
- **Statistical** — autocorrelation, relative strength vs. industry/market
- **Pairs trading** — cointegration-style relative pricing signals

### 2. Signal validation — before a factor earns a place in the model
Rather than throwing every factor into a GBM and letting feature importance sort it out, each signal's forecasting power is tested independently:
- **Rank Information Coefficient (Spearman)** between each factor and forward 5-day returns, computed cross-sectionally per date
- **Newey-West HAC-adjusted t-tests** on the IC time series to account for serial correlation in overlapping return windows
- **Benjamini-Hochberg FDR correction** across all 40+ simultaneous significance tests, to control the false discovery rate rather than relying on uncorrected p-values (a common source of false "alpha" in factor research)

### 3. Model training & validation
Three tree-based architectures (Decision Tree, LightGBM, XGBoost) plus a GRU and linear baseline, each tuned via randomised hyperparameter search. Validation uses a **custom purged walk-forward validator**:
- Weekly rebalancing with an expanding training window
- A **purge/embargo gap** equal to the forecast horizon (5 days) between train and test cuts, preventing label leakage from overlapping forward-return targets — the same failure mode that inflates backtested Sharpe ratios in a lot of published retail quant work
- Separate feature matrices at stock-level, market-level, and macro+market-level, to isolate where predictive power is actually coming from

### 4. Model comparison — formal hypothesis testing, not just "which curve is higher"
- **Paired t-tests** on mean weekly IC to test whether one model architecture significantly outperforms another
- **McNemar's test** on the hit-rate contingency table to test whether two models disagree on directional calls more than chance would suggest
- Every comparison in the dashboard is backed by a stated null hypothesis, test statistic, and p-value — not a visual read of an equity curve

### 5. Portfolio construction
- Top-20 / bottom-20 cross-sectional selection by model-predicted forward return each rebalance
- **Mean-variance optimisation** (SciPy SLSQP) on each side of the book independently, with position bounds, a ridge-regularised covariance matrix (to keep the optimiser numerically stable on a rolling 63-day covariance window), and a Sharpe-maximising objective
- Backtest metrics: annualised return, Sharpe, Sortino, Calmar, max drawdown, win rate — computed from realised weekly portfolio returns, not in-sample fitted values

### Result
**Out-of-sample Sharpe ratio: 1.50**, walk-forward validated with purging/embargo and FDR-corrected signal selection.

---

## Dashboard

A Streamlit application (containerised with Docker, deployed on AWS) covering:
- **Portfolio** — current long/short book, weights, sector exposure
- **Backtest Performance** — equity curve, drawdown, return distribution, rolling metrics
- **Model Comparison** — side-by-side IC, hit rate, and the hypothesis test results above
- **Methodology & Lessons** — a plain-language walkthrough of the validation design and what didn't work

## Tech stack

`Python` · `pandas` / `numpy` / `pyarrow` (Parquet-backed data layer) · `scikit-learn`, `XGBoost`, `LightGBM` · `statsmodels` (HAC t-tests, FDR correction) · `SciPy` (portfolio optimisation) · `arch` (GARCH volatility modelling) · `Streamlit` + `Docker`, deployed on `AWS` · `yfinance` for price data, RBA/ABS for macro series

---

## Honest limitations

In the interest of not overselling this the way I'd be uncomfortable defending in an interview:
- Universe is ASX-listed equities only — no fixed income, FX, or multi-asset exposure yet
- Backtest costs (slippage, market impact) are simplified relative to live execution reality
- Effective statistical breadth is materially lower than the raw factor count suggests once cross-sectional correlation between signals is accounted for — the FDR correction step exists specifically to guard against overstating this
- Some data files and model artifacts are currently committed to the repo rather than tracked via DVC/external storage — a packaging cleanup in progress

## Running it locally

```bash
pip install -r requirements.txt
python main.py                    # run the data ingestion pipeline
python scripts/run_fetch.py       # fetch/refresh raw price & macro data
python scripts/main_models.py     # train and validate models
python scripts/run_portfolio.py   # build the long-short portfolio
streamlit run app/app.py          # launch the dashboard
```

---

*Built and maintained by Gordon — a self-directed project alongside full-time work, motivated by a long-standing interest in applied ML and systematic investing.*