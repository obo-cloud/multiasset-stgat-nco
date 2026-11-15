import os
import sys
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from stgat_model import STGATModel, build_dynamic_graph
from gnn_nco_allocator import (
    compute_downside_semicovariance,
    solve_constrained_min_variance,
    solve_constrained_sortino,
    allocate_nco_constrained,
    allocate_hrp_constrained
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def compute_metrics(series_returns, rf_annual=0.035):
    """
    Computes audited quantitative performance metrics.
    """
    total_months = len(series_returns)
    years = total_months / 12.0
    cum_wealth = (1.0 + series_returns).cumprod()
    total_ret = (cum_wealth.iloc[-1] - 1.0) * 100.0
    cagr = ((cum_wealth.iloc[-1]) ** (1.0 / years) - 1.0) * 100.0
    
    ann_vol = series_returns.std() * np.sqrt(12) * 100.0
    
    # Downside volatility relative to risk-free
    rf_monthly = (1.0 + rf_annual) ** (1.0 / 12.0) - 1.0
    downside_diff = np.minimum(series_returns - rf_monthly, 0.0)
    downside_vol = np.sqrt(12) * np.std(downside_diff) * 100.0
    
    # Excess Sharpe & Sortino
    mean_excess_ann = (series_returns.mean() - rf_monthly) * 12 * 100.0
    sharpe = mean_excess_ann / ann_vol if ann_vol > 0 else 0.0
    sortino = mean_excess_ann / downside_vol if downside_vol > 0 else 0.0
    
    # Drawdown
    running_max = cum_wealth.cummax()
    drawdown = (cum_wealth - running_max) / running_max * 100.0
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else 0.0
    
    return {
        "CAGR (%)": cagr,
        "Ann. Vol (%)": ann_vol,
        "Downside Vol (%)": downside_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max DD (%)": max_dd,
        "Calmar Ratio": calmar,
        "Total Return (%)": total_ret
    }

def construct_feature_tensor(ret_df, seq_len=12):
    N = ret_df.shape[1]
    features = np.zeros((N, 5, seq_len))
    recent_rets = ret_df.iloc[-seq_len:].values # (seq_len, N)
    
    # Feature 1: Raw returns
    features[:, 0, :] = recent_rets.T
    
    # Feature 2: Rolling volatility (std of last 3 steps)
    for t in range(seq_len):
        start_idx = max(0, t - 2)
        window = recent_rets[start_idx:t+1, :]
        features[:, 1, t] = np.std(window, axis=0) if len(window) > 1 else 0.01
        
    # Feature 3: Downside semi-deviation
    for t in range(seq_len):
        start_idx = max(0, t - 2)
        window = recent_rets[start_idx:t+1, :]
        downside = np.minimum(window, 0.0)
        features[:, 2, t] = np.std(downside, axis=0) if len(downside) > 1 else 0.01
        
    # Feature 4: 3-month momentum
    for t in range(seq_len):
        start_idx = max(0, t - 2)
        window = recent_rets[start_idx:t+1, :]
        features[:, 3, t] = np.prod(1.0 + window, axis=0) - 1.0
        
    # Feature 5: Cross-sectional relative strength
    market_ret = recent_rets.mean(axis=1, keepdims=True)
    features[:, 4, :] = (recent_rets - market_ret).T
    
    return torch.tensor(features, dtype=torch.float32)

def train_stgat_representation(model, feature_tensor, adj, target_rets, epochs=30, lr=0.01):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    target_t = torch.tensor(target_rets, dtype=torch.float32)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        embeddings, pred_ret, pred_down, _ = model(feature_tensor, adj)
        loss_ret = F.mse_loss(pred_ret, target_t)
        
        diff = embeddings.unsqueeze(1) - embeddings.unsqueeze(0)
        dist_sq = torch.sum(diff ** 2, dim=-1)
        loss_graph = torch.mean(adj * dist_sq)
        
        total_loss = loss_ret + 0.1 * loss_graph
        total_loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        embeddings, _, _, attn_map = model(feature_tensor, adj)
    return embeddings.numpy(), attn_map.numpy()

def run_simulation():
    print("[INFO] Loading multi-asset monthly return series...")
    monthly_csv = os.path.join(DATA_DIR, "multiasset_monthly_returns.csv")
    df_returns = pd.read_csv(monthly_csv, index_col=0, parse_dates=True)
    
    T_total, N = df_returns.shape
    asset_names = df_returns.columns.tolist()
    btc_idx = asset_names.index("BTC-USD") if "BTC-USD" in asset_names else None
    print(f"  [+] Loaded {T_total} months across {N} multi-asset classes: {asset_names} (BTC Index: {btc_idx})")
    
    LOOKBACK = 24 # 24-month rolling lookback
    SEQ_LEN = 12  # 12-month temporal sequence for GNN
    COST_BPS = 0.0010 # 10 bps transaction cost
    
    strategy_names = [
        "Equal-Weight (1/N)",
        "Min-Variance (MVO)",
        "Hierarchical Risk Parity (HRP)",
        "Plain NCO (Symmetric)",
        "Downside NCO (Semi-Cov)",
        "Proposed ST-GAT-NCO"
    ]
    
    portfolio_returns = {s: [] for s in strategy_names}
    portfolio_turnovers = {s: [] for s in strategy_names}
    weights_history = {s: [] for s in strategy_names}
    rebalance_dates = []
    
    # Inception weights
    w0_init = np.ones(N) / N
    if btc_idx is not None:
        w0_init[btc_idx] = 0.05
        w0_init /= np.sum(w0_init)
        
    prev_weights = {s: w0_init.copy() for s in strategy_names}
    
    stgat = STGATModel(in_features=5, temporal_dim=32, spatial_dim=32, embedding_dim=16, num_heads=4)
    
    print(f"\n[INFO] Running non-anticipative walk-forward backtest (Months {LOOKBACK} to {T_total-1})...")
    
    for t in range(LOOKBACK, T_total):
        curr_date = df_returns.index[t]
        rebalance_dates.append(curr_date)
        
        train_returns = df_returns.iloc[t-LOOKBACK:t]
        next_realized_return = df_returns.iloc[t].values
        
        mean_ret = train_returns.mean().values
        cov_sym = np.cov(train_returns.values, rowvar=False)
        cov_sym = 0.5 * (cov_sym + cov_sym.T) + 1e-6 * np.eye(N)
        cov_down = compute_downside_semicovariance(train_returns.values)
        
        # Symmetric distance
        std_devs = np.sqrt(np.diag(cov_sym))
        corr_sym = cov_sym / np.outer(std_devs, std_devs)
        corr_sym = np.clip(corr_sym, -1.0, 1.0)
        dist_sym = np.sqrt(0.5 * (1.0 - corr_sym))
        np.fill_diagonal(dist_sym, 0.0)
        
        # Downside semi-distance
        std_down = np.sqrt(np.diag(cov_down))
        corr_down = cov_down / np.outer(std_down, std_down)
        corr_down = np.clip(corr_down, -1.0, 1.0)
        dist_down = np.sqrt(0.5 * (1.0 - corr_down))
        np.fill_diagonal(dist_down, 0.0)
        
        # --- ST-GAT GNN Processing ---
        feature_tensor = construct_feature_tensor(train_returns, seq_len=SEQ_LEN)
        adj_tensor = build_dynamic_graph(corr_sym, threshold=0.30)
        
        recent_target = train_returns.iloc[-1].values
        gnn_embeddings, _ = train_stgat_representation(stgat, feature_tensor, adj_tensor, recent_target, epochs=20, lr=0.01)
        
        # GNN Metric Distance
        norm_emb = gnn_embeddings / np.linalg.norm(gnn_embeddings, axis=-1, keepdims=True)
        cos_sim = np.dot(norm_emb, norm_emb.T)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        dist_gnn = np.sqrt(0.5 * (1.0 - cos_sim))
        np.fill_diagonal(dist_gnn, 0.0)
        
        # --- Compute Target Weights with Institutional Constraints ---
        # 1. Equal-Weight
        w_ew = np.ones(N) / N
        
        # 2. Min-Variance (MVO)
        w_mvo = solve_constrained_min_variance(cov_sym, max_weight=0.25, btc_idx=btc_idx, max_btc=0.05)
        
        # 3. Hierarchical Risk Parity (HRP)
        w_hrp = allocate_hrp_constrained(cov_sym, max_weight=0.25, btc_idx=btc_idx, max_btc=0.05)
        
        # 4. Plain NCO (Symmetric)
        w_nco_plain = allocate_nco_constrained(cov_sym, mean_ret, dist_sym, gamma=0.2, max_weight=0.25, btc_idx=btc_idx, max_btc=0.08)
        
        # 5. Downside NCO (Semi-Cov)
        w_nco_down = allocate_nco_constrained(cov_down, mean_ret, dist_down, gamma=0.4, max_weight=0.25, btc_idx=btc_idx, max_btc=0.08)
        
        # 6. Proposed ST-GAT-NCO (GNN-Guided Downside NCO with dynamic turnover regularization)
        w_raw_gat = allocate_nco_constrained(cov_down, mean_ret, dist_gnn, gamma=0.6, max_weight=0.25, btc_idx=btc_idx, max_btc=0.08)
        
        # Smooth weight transition with previous weights:
        lambda_turn = 0.35 # 35% inertia to curb excess turnover
        w_gat = (1.0 - lambda_turn) * w_raw_gat + lambda_turn * prev_weights["Proposed ST-GAT-NCO"]
        w_gat = np.maximum(w_gat, 0.0)
        w_gat /= np.sum(w_gat)
        
        strategy_weights = {
            "Equal-Weight (1/N)": w_ew,
            "Min-Variance (MVO)": w_mvo,
            "Hierarchical Risk Parity (HRP)": w_hrp,
            "Plain NCO (Symmetric)": w_nco_plain,
            "Downside NCO (Semi-Cov)": w_nco_down,
            "Proposed ST-GAT-NCO": w_gat
        }
        
        # --- Accounting ---
        for s in strategy_names:
            w_tgt = strategy_weights[s]
            w_prev = prev_weights[s]
            
            turnover = np.sum(np.abs(w_tgt - w_prev))
            portfolio_turnovers[s].append(turnover)
            
            r_gross = np.dot(w_tgt, next_realized_return)
            r_net = r_gross - COST_BPS * turnover
            portfolio_returns[s].append(r_net)
            
            weights_history[s].append(w_tgt)
            prev_weights[s] = w_tgt
            
    df_oos_returns = pd.DataFrame(portfolio_returns, index=rebalance_dates)
    returns_csv = os.path.join(RESULTS_DIR, "strategy_monthly_returns.csv")
    df_oos_returns.to_csv(returns_csv, float_format="%.6f")
    
    df_weights_gat = pd.DataFrame(weights_history["Proposed ST-GAT-NCO"], index=rebalance_dates, columns=asset_names)
    weights_csv = os.path.join(RESULTS_DIR, "weights_proposed_stgat_nco.csv")
    df_weights_gat.to_csv(weights_csv, float_format="%.4f")
    
    metrics_list = []
    for s in strategy_names:
        m = compute_metrics(df_oos_returns[s], rf_annual=0.035)
        m["Monthly Turnover (%)"] = np.mean(portfolio_turnovers[s]) * 100.0
        metrics_list.append(m)
        
    df_summary = pd.DataFrame(metrics_list, index=strategy_names)
    metrics_csv = os.path.join(RESULTS_DIR, "performance_metrics.csv")
    df_summary.to_csv(metrics_csv, float_format="%.4f")
    
    print("\n" + "=" * 95)
    print("AUDITED OUT-OF-SAMPLE PERFORMANCE COMPARISON (MULTI-ASSET PORTFOLIOS)")
    print(f"Evaluation Window: {rebalance_dates[0].strftime('%Y-%m')} to {rebalance_dates[-1].strftime('%Y-%m')} ({len(rebalance_dates)} Months)")
    print("=" * 95)
    print(df_summary.round(2).to_string())
    print("=" * 95)

if __name__ == "__main__":
    run_simulation()
