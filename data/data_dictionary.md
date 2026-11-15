# Data Dictionary: Global Multi-Asset Universe

This dataset provides historical daily adjusted closing prices, daily returns, and monthly simple returns across 13 heterogeneous global asset classes from January 2018 to January 2026 (97 months).

## Temporal Coverage
- **Total Period:** January 2018 – January 2026 (97 monthly periods)
- **Warmup / Initial Estimation Window:** January 2018 – December 2019 (24 months)
- **Walk-Forward Out-of-Sample Horizon:** January 2020 – January 2026 (73 months)

## Asset Classes & Identifiers

| Ticker | Asset Class | Underlying Index / Exposure | Provider | Currency | Role in Portfolio |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SPY** | Equities | S&P 500 US Large Cap ETF | Yahoo Finance | USD | Core equity growth engine |
| **QQQ** | Equities | Invesco QQQ Trust (Nasdaq 100 Tech ETF) | Yahoo Finance | USD | High-growth technology growth driver |
| **EEM** | Equities | iShares MSCI Emerging Markets ETF | Yahoo Finance | USD | International emerging markets diversifier |
| **IEF** | Fixed Income | iShares 7-10 Year Treasury Bond ETF | Yahoo Finance | USD | Intermediate sovereign duration hedge |
| **TLT** | Fixed Income | iShares 20+ Year Treasury Bond ETF | Yahoo Finance | USD | Long-duration crisis flight-to-safety asset |
| **HYG** | Fixed Income | iShares iBoxx $ High Yield Corporate Bond ETF | Yahoo Finance | USD | Credit spread & income yield capture |
| **LQD** | Fixed Income | iShares iBoxx $ Investment Grade Corporate Bond ETF | Yahoo Finance | USD | Quality institutional credit reserve |
| **GLD** | Commodities | SPDR Gold Shares ETF (Physical Gold) | Yahoo Finance | USD | Safe-haven inflation & geopolitical hedge |
| **DBC** | Commodities | Invesco DB Commodity Index Tracking Fund ETF | Yahoo Finance | USD | Broad commodity cyclical inflation hedge |
| **VNQ** | Real Estate | Vanguard Real Estate ETF (US REITs) | Yahoo Finance | USD | Real asset income & inflation pass-through |
| **UUP** | Currency | Invesco DB US Dollar Index Bullish Fund ETF | Yahoo Finance | USD | Global liquidity squeeze & USD buffer |
| **SHV** | Cash Proxy | iShares Short Treasury Bond ETF (0-1 Year) | Yahoo Finance | USD | Risk-free cash preservation reserve |
| **BTC-USD** | Digital Asset | Bitcoin Spot Price | Yahoo Finance | USD | Non-correlated convex growth diversifier |

## Macroeconomic Regime Benchmarks
- **`^VIX`**: CBOE Volatility Index (Equity market fear / uncertainty gauge)
- **`^IRX`**: 13-Week US Treasury Bill Yield (Short-term risk-free rate proxy)

## Modeling & Accounting Conventions
1. **Portfolio Net Asset Value (NAV):**
   - Simple monthly returns: $R_{p,t} = \mathbf{w}_t^\top \mathbf{R}_t - c \times \text{TO}_t$
   - Proportional transaction cost: $c = 10\text{ bps}$ ($0.10\%$) applied to total monthly turnover.
2. **Dynamic Graph Construction:**
   - Adjacency matrix $\mathbf{A}_t$ is computed via thresholded exponential correlation distance over a 12-month rolling window.
   - Node feature tensor $\mathbf{X}_t \in \mathbb{R}^{N \times F \times T}$ captures multi-horizon returns (1m, 3m, 6m, 12m) and rolling downside semi-volatility.
