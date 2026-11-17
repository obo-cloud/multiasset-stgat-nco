# Dynamic Spatio-Temporal Graph Attention Networks with Nested Clustered Optimization for Multi-Asset Portfolios

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2611.19903-b31b1b.svg)](https://arxiv.org/abs/2611.19903)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2611.19903-blue.svg)](https://doi.org/10.48550/arXiv.2611.19903)

Official open-source data and code repository for the research paper:  
**"Dynamic Spatio-Temporal Graph Attention Networks with Nested Clustered Optimization for Multi-Asset Portfolios"**  
*Author:* **Supriadi** (`supriadi@coconut.or.id`)  
*Target:* **IEEE Transactions on Neural Networks and Learning Systems (Preprint)**

---

## Overview

This repository provides an end-to-end reproducible quantitative finance and deep learning pipeline implementing **Dynamic Spatio-Temporal Graph Attention Networks with Nested Clustered Optimization (ST-GAT-NCO)** for tactical multi-asset allocation.

Traditional portfolio optimization methods, such as Markowitz Mean-Variance Optimization (MVO), suffer from acute covariance ill-conditioning. While hierarchical clustering techniques (e.g., Hierarchical Risk Parity and classical Nested Clustered Optimization) insulate portfolios from covariance noise, they rely on static pairwise Pearson correlation metrics that fail to capture time-varying, asymmetric downside risk spillovers across heterogeneous asset classes.

### Key Innovations
1. **Causal Temporal Dynamics:** Extracts multi-horizon momentum and downside volatility via causal dilated 1D-convolutions without lookahead leakage.
2. **Multi-Head Graph Attention:** Dynamically learns continuous spatial node representations and topological cross-asset risk dependencies over time-varying market graphs.
3. **GNN-Guided Downside NCO:** Derives a non-Euclidean angular metric distance space from learned graph embeddings to direct two-stage downside semi-variance optimization.
4. **Institutional Execution & Friction Modeling:** Incorporates convex quadratic turnover smoothing and evaluates performance net of 10 bps transaction slippage over a 73-month walk-forward horizon.

---

## Repository Structure

```text
multiasset-stgat-nco/
├── data/
│   ├── multiasset_monthly_returns.csv     # Monthly returns for walk-forward allocation (13 assets, 97 months)
│   ├── multiasset_daily_returns.csv       # Daily return series
│   ├── multiasset_daily_prices.csv        # Historical daily adjusted close prices (2018–2026)
│   ├── macro_monthly_levels.csv           # Macroeconomic regime series (CBOE VIX, 13-week T-Bill)
│   └── data_dictionary.md                 # Data sources, proxy identifiers, and accounting definitions
├── figures/                               # Publication-grade figures (.png)
│   ├── fig_gnn_framework.png              # End-to-end ST-GAT-NCO architectural pipeline
│   ├── fig1_cumulative_wealth.png         # Out-of-sample cumulative wealth trajectories (log-scale)
│   ├── fig2_drawdowns.png                 # Underwater drawdown comparisons
│   ├── fig3_attention_graph.png           # Learned multi-head graph attention topology
│   ├── fig4_dendrogram_embeddings.png     # Hierarchical dendrogram in latent GNN embedding space
│   ├── fig5_weights_allocation.png        # Dynamic asset allocation area plot across macro regimes
│   └── fig6_performance_comparison.png   # Risk-adjusted performance bar chart summary
├── results/                               # Tabular backtest outputs and transaction ledgers
│   ├── performance_metrics.csv            # Audited summary risk-adjusted metrics across all benchmarks
│   ├── strategy_monthly_returns.csv       # Realized monthly out-of-sample return series
│   └── weights_proposed_stgat_nco.csv     # Dynamic allocation weight ledger (2020–2026)
├── src/                                   # Core model and backtesting scripts
│   ├── download_data.py                   # Automated Yahoo Finance market data fetcher
│   ├── stgat_model.py                     # PyTorch ST-GAT deep neural network architecture
│   ├── gnn_nco_allocator.py               # Downside NCO solver and GNN metric distance functions
│   ├── run_backtest.py                    # Non-anticipative walk-forward simulation engine
│   └── plot_figures.py                    # Publication figure generator
├── requirements.txt                       # Python dependencies
├── LICENSE                                # MIT Open Source License
└── README.md                              # Project documentation
```

---

## Audited Empirical Results (2020–2026 Out-of-Sample, 73 Months)

All metrics are evaluated net of 10 bps transaction costs with an annualized risk-free rate $r_f = 3.50\%$:

| Strategy | CAGR (%) | Ann. Vol (%) | Downside Vol (%) | Sharpe Ratio | Sortino Ratio | Max DD (%) | Calmar Ratio | Total Ret (%) | Monthly Turnover (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Equal-Weight ($1/N$)** | 11.40 | 11.04 | 6.11 | 0.72 | 1.31 | -19.44 | 0.59 | 92.84 | 0.07 |
| **Min-Variance (MVO)** | 3.68 | 2.62 | 1.63 | 0.08 | 0.13 | -3.24 | 1.13 | 24.56 | 6.81 |
| **Hierarchical Risk Parity (HRP)** | 2.65 | 0.73 | 0.52 | -1.14 | -1.59 | -0.53 | 4.96 | 17.21 | 4.61 |
| **Plain NCO (Symmetric)** | 4.11 | 4.58 | 2.65 | 0.15 | 0.26 | -8.58 | 0.48 | 27.75 | 22.40 |
| **Downside NCO (Semi-Cov)** | 4.26 | 5.74 | 3.59 | 0.16 | 0.25 | -16.36 | 0.26 | 28.92 | 20.91 |
| **Proposed ST-GAT-NCO** | **5.32** | **6.61** | **4.07** | **0.30** | **0.48** | **-13.07** | **0.41** | **37.06** | **30.74** |

*Note: Monthly turnover is reported on a two-way basis (30.74%/mo), which corresponds to 15.31%/mo on a one-way basis.*

---

## Quickstart & Reproduction

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/obo-cloud/multiasset-stgat-nco.git
cd multiasset-stgat-nco
pip install -r requirements.txt
```

### 2. Fetch Historical Data (Optional)
The repository already includes pre-processed datasets in `data/`. To refresh data directly from Yahoo Finance:
```bash
python src/download_data.py
```

### 3. Run Walk-Forward Backtesting
Execute the 73-month walk-forward simulation across all six strategies:
```bash
python src/run_backtest.py
```
This script computes all performance metrics and saves summary tables in `results/`.

### 4. Generate Publication Figures
Generate all visual figures and save them to `figures/`:
```bash
python src/plot_figures.py
```

---

## Citation

If you find this code or research useful in your work, please cite:

```bibtex
@article{supriadi2026stgat,
  author  = {Supriadi},
  title   = {Dynamic Spatio-Temporal Graph Attention Networks with Nested Clustered Optimization for Multi-Asset Portfolios},
  journal = {arXiv preprint arXiv:2611.19903 [q-fin.PM]},
  year    = {2026},
  url     = {https://arxiv.org/abs/2611.19903},
  doi     = {10.48550/arXiv.2611.19903}
}
```

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
