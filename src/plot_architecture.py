import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

def draw_architecture():
    # Large, crisp canvas with high resolution
    fig, ax = plt.subplots(figsize=(15.5, 8.2), dpi=300)
    ax.set_xlim(0, 155)
    ax.set_ylim(0, 82)
    ax.axis("off")

    # High-Contrast Academic Palette
    c_bg_group = "#fafbfc"
    c_border_group = "#95a5a6"
    
    c_input_bg = "#e8f4f8"
    c_input_edge = "#0288d1"
    
    c_temp_bg = "#e1f5fe"
    c_temp_edge = "#0277bd"
    
    c_top_bg = "#f3e5f5"
    c_top_edge = "#7b1fa2"
    
    c_gat_bg = "#ede7f6"
    c_gat_edge = "#4527a0"
    
    c_metric_bg = "#fff8e1"
    c_metric_edge = "#f57c00"
    
    c_nco_bg = "#e8f5e9"
    c_nco_edge = "#2e7d32"
    
    c_out_bg = "#fbe9e7"
    c_out_edge = "#d84315"

    font_sec = {"fontsize": 10.5, "fontweight": "bold"}
    font_title = {"fontsize": 10.5, "fontweight": "bold", "color": "#212529"}
    font_sub = {"fontsize": 9.0, "fontweight": "bold", "color": "#37474f"}

    # -------------------------------------------------------------
    # 4 High-Level Enclosures (Phases)
    # -------------------------------------------------------------
    # Phase I: Spatio-Temporal Graph Encoder (x: 2 -> 82, y: 3 -> 78)
    box_p1 = FancyBboxPatch((2, 3), 80, 75, boxstyle="round,pad=1.2",
                            fc=c_bg_group, ec=c_border_group, lw=1.2, ls="--")
    ax.add_patch(box_p1)
    ax.text(42, 75.2, "PHASE I: DYNAMIC SPATIO-TEMPORAL GRAPH ENCODER (ST-GAT)", 
            ha="center", va="center", color="#1a237e", **font_sec)

    # Phase II: Latent Metric Geometry (x: 84 -> 116, y: 41 -> 78)
    box_p2 = FancyBboxPatch((84, 41), 32, 37, boxstyle="round,pad=1.2",
                            fc=c_bg_group, ec=c_border_group, lw=1.2, ls="--")
    ax.add_patch(box_p2)
    ax.text(100, 75.2, "PHASE II: METRIC GEOMETRY", 
            ha="center", va="center", color="#e65100", **font_sec)

    # Phase III: Two-Stage Downside NCO (x: 118 -> 153, y: 3 -> 78)
    box_p3 = FancyBboxPatch((118, 3), 35, 75, boxstyle="round,pad=1.2",
                            fc=c_bg_group, ec=c_border_group, lw=1.2, ls="--")
    ax.add_patch(box_p3)
    ax.text(135.5, 75.2, "PHASE III: DOWNSIDE NCO OPTIMIZER", 
            ha="center", va="center", color="#1b5e20", **font_sec)

    # Phase IV: Execution & Turnover Control (x: 84 -> 116, y: 3 -> 39)
    box_p4 = FancyBboxPatch((84, 3), 32, 36, boxstyle="round,pad=1.2",
                            fc=c_bg_group, ec=c_border_group, lw=1.2, ls="--")
    ax.add_patch(box_p4)
    ax.text(100, 36.2, "PHASE IV: TACTICAL REBALANCING", 
            ha="center", va="center", color="#bf360c", **font_sec)

    # -------------------------------------------------------------
    # Phase I Inner Modules
    # -------------------------------------------------------------
    # 1. Input Tensor
    box_in = FancyBboxPatch((4, 43), 18, 26, boxstyle="round,pad=0.8",
                            fc=c_input_bg, ec=c_input_edge, lw=1.8)
    ax.add_patch(box_in)
    ax.text(13, 65.5, "Multi-Asset\nInput Tensor", ha="center", va="center", **font_title)
    ax.text(13, 57.5, r"$\mathbf{X}_t \in \mathbb{R}^{N \times F \times K}$", ha="center", va="center", 
            fontsize=9.2, fontweight="bold", color="#01579b")
    desc_in = (
        "• N = 13 Global Assets\n"
        "• K = 12 Rolling Months\n"
        "• F = 5 Signal Features:\n"
        "   1. Monthly Return\n"
        "   2. Realized Volatility\n"
        "   3. Downside Semi-Dev\n"
        "   4. 3M/12M Momentum\n"
        "   5. Relative Strength"
    )
    ax.text(5.5, 49.5, desc_in, ha="left", va="center", fontsize=7.2, color="#263238")

    # 2. Causal 1D TCN
    box_tcn = FancyBboxPatch((25, 43), 19, 26, boxstyle="round,pad=0.8",
                             fc=c_temp_bg, ec=c_temp_edge, lw=1.8)
    ax.add_patch(box_tcn)
    ax.text(34.5, 65.5, "Causal Dilated\nTemporal Conv", ha="center", va="center", **font_title)
    ax.text(34.5, 57.5, r"$\mathbf{h}_i^{(0)} \in \mathbb{R}^{d_{\mathrm{temp}}}$", ha="center", va="center", 
            fontsize=9.2, fontweight="bold", color="#01579b")
    desc_tcn = (
        "• Dilated 1D Kernels\n"
        "• Causal Zero-Padding\n"
        "• LeakyReLU Activation\n"
        "• Residual Skip Highway\n"
        "• Extracts Asset Temporal\n"
        "  Momentum & Volatility\n"
        "  Memory without leak"
    )
    ax.text(26.5, 49.5, desc_tcn, ha="left", va="center", fontsize=7.2, color="#263238")

    # 3. Dynamic Topology Construction
    box_top = FancyBboxPatch((4, 7), 40, 31, boxstyle="round,pad=0.8",
                             fc=c_top_bg, ec=c_top_edge, lw=1.8)
    ax.add_patch(box_top)
    ax.text(24, 34.5, "Dynamic Graph Topology Construction", ha="center", va="center", **font_title)
    ax.text(24, 29.5, r"$\mathcal{G}_t = (\mathcal{V}, \mathcal{E}_t, \mathbf{A}_t), \quad A_{ij,t} = \mathbf{1}_{|C_{ij,t}| \geq \tau \ \mathrm{or} \ i=j}$", 
            ha="center", va="center", fontsize=7.8, fontweight="bold", color="#4a148c")
    desc_top = (
        "• Rolling Lookback L = 24 Months\n"
        "• Connectivity Sparsity Threshold: tau = 0.30\n"
        "• Dynamic Multi-Asset Adjacency Matrix with Self-Loops\n"
        "• Models Systemic Contagion & Flight-to-Safety Paths"
    )
    ax.text(5.5, 18.5, desc_top, ha="left", va="center", fontsize=7.3, color="#263238")

    # Mini graph schematic inside Topology
    circle_style = dict(boxstyle="circle,pad=0.25", ec="#4a148c", lw=1.2)
    node_coords = [(11, 10), (18, 12), (28, 12), (36, 10), (23, 7)]
    node_lbls = ["SPY", "QQQ", "GLD", "TLT", "BTC"]
    for (nx, ny), nlbl in zip(node_coords, node_lbls):
        ax.add_patch(FancyBboxPatch((nx-1.8, ny-1.5), 3.6, 3.0, boxstyle="round,pad=0.2", fc="#ffffff", ec="#6a1b9a", lw=1.2))
        ax.text(nx, ny, nlbl, ha="center", va="center", fontsize=6.2, fontweight="bold", color="#4a148c")
    # Edges
    ax.plot([11, 18], [10, 12], color="#8e24aa", lw=1.2)
    ax.plot([18, 28], [12, 12], color="#8e24aa", lw=1.2, ls=":")
    ax.plot([28, 36], [12, 10], color="#8e24aa", lw=1.2)
    ax.plot([18, 23], [12, 7], color="#8e24aa", lw=1.0, ls="--")

    # 4. Multi-Head Graph Attention Network (GAT Architecture Detail)
    box_gat = FancyBboxPatch((47, 7), 33, 62, boxstyle="round,pad=0.8",
                             fc=c_gat_bg, ec=c_gat_edge, lw=2.0)
    ax.add_patch(box_gat)
    ax.text(63.5, 65.5, "Multi-Head Spatial\nGraph Attention Layer", ha="center", va="center", **font_title)
    
    # Internal GAT Mechanism Schematic
    # Node i and Node j projection
    ax.add_patch(FancyBboxPatch((50, 52), 9, 5.5, boxstyle="round,pad=0.3", fc="#ffffff", ec="#5e35b1", lw=1.3))
    ax.text(54.5, 54.7, r"$\mathbf{W}^{(k)}\mathbf{h}_i$", ha="center", va="center", fontsize=7.2, fontweight="bold", color="#311b92")

    ax.add_patch(FancyBboxPatch((67, 52), 9, 5.5, boxstyle="round,pad=0.3", fc="#ffffff", ec="#5e35b1", lw=1.3))
    ax.text(71.5, 54.7, r"$\mathbf{W}^{(k)}\mathbf{h}_j$", ha="center", va="center", fontsize=7.2, fontweight="bold", color="#311b92")

    # Concat block
    ax.add_patch(FancyBboxPatch((55, 43.5), 17, 5.2, boxstyle="round,pad=0.3", fc="#d1c4e9", ec="#512da8", lw=1.2))
    ax.text(63.5, 46.1, r"$[\mathbf{W}^{(k)}\mathbf{h}_i \,\|\, \mathbf{W}^{(k)}\mathbf{h}_j]$", 
            ha="center", va="center", fontsize=7.2, fontweight="bold", color="#1a237e")

    # Attention vector projection
    ax.add_patch(FancyBboxPatch((53, 34), 21, 6.2, boxstyle="round,pad=0.3", fc="#ffffff", ec="#311b92", lw=1.4))
    ax.text(63.5, 37.8, r"$\alpha_{ij}^{(k)} = \mathrm{Softmax}_j \left( \mathrm{LeakyReLU}(\mathbf{a}_k^T [\dots]) \right)$", 
            ha="center", va="center", fontsize=6.8, fontweight="bold", color="#311b92")
    ax.text(63.5, 35.0, "Attention Weight across Edges", ha="center", va="center", fontsize=6.0, color="#5e35b1")

    # Multi-head aggregation formula
    ax.add_patch(FancyBboxPatch((49, 21), 29, 9.5, boxstyle="round,pad=0.4", fc="#ffffff", ec="#4527a0", lw=1.5))
    ax.text(63.5, 27.5, "Multi-Head Feature Aggregation", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#1a237e")
    ax.text(63.5, 23.2, r"$\mathbf{z}_i = \sigma \left( \sum_{k=1}^{K_h} \sum_{j \in \mathcal{N}_i} \alpha_{ij}^{(k)} \mathbf{W}^{(k)} \mathbf{h}_j \right)$", 
            ha="center", va="center", fontsize=7.5, fontweight="bold", color="#1a237e")

    desc_gat = (
        "• Multi-Head: K_h = 2 Attentive Heads\n"
        "• Shared Attention Weight Vector a_k\n"
        "• Graph Laplacian Smoothing Regularization\n"
        "• Dynamic Node Embeddings z_i in R^d"
    )
    ax.text(49, 13.5, desc_gat, ha="left", va="center", fontsize=7.0, color="#263238")

    # -------------------------------------------------------------
    # Phase II: Latent Metric Geometry
    # -------------------------------------------------------------
    box_dist = FancyBboxPatch((86, 44), 28, 25, boxstyle="round,pad=0.8",
                              fc=c_metric_bg, ec=c_metric_edge, lw=1.8)
    ax.add_patch(box_dist)
    ax.text(100, 65.5, "Latent Embedding &\nMetric Space", ha="center", va="center", **font_title)
    
    # Hypersphere norm
    ax.add_patch(FancyBboxPatch((88, 54), 24, 6.0, boxstyle="round,pad=0.3", fc="#ffffff", ec="#ffa000", lw=1.2))
    ax.text(100, 57.0, r"$\tilde{\mathbf{z}}_i = \frac{\mathbf{z}_i}{\|\mathbf{z}_i\|_2} \in \mathbb{S}^{d-1}$", 
            ha="center", va="center", fontsize=8.2, fontweight="bold", color="#e65100")

    # Angular distance
    ax.add_patch(FancyBboxPatch((88, 46), 24, 6.5, boxstyle="round,pad=0.3", fc="#ffffff", ec="#e65100", lw=1.4))
    ax.text(100, 50.2, r"$D_{ij}^{\mathrm{GNN}} = \sqrt{\frac{1}{2} (1 - \tilde{\mathbf{z}}_i^T \tilde{\mathbf{z}}_j)}$", 
            ha="center", va="center", fontsize=8.2, fontweight="bold", color="#b71c1c")
    ax.text(100, 47.0, "Non-Euclidean Distance Space", ha="center", va="center", fontsize=6.2, fontstyle="italic", color="#795548")

    # -------------------------------------------------------------
    # Phase III: Two-Stage Downside NCO Optimizer
    # -------------------------------------------------------------
    # Stage 1: Intra-Cluster
    box_stg1 = FancyBboxPatch((120, 44), 31, 25, boxstyle="round,pad=0.8",
                              fc=c_nco_bg, ec=c_nco_edge, lw=1.8)
    ax.add_patch(box_stg1)
    ax.text(135.5, 65.5, "Stage 1: Intra-Cluster\nDownside Optimization", ha="center", va="center", **font_title)
    
    ax.add_patch(FancyBboxPatch((122, 54), 27, 6.5, boxstyle="round,pad=0.3", fc="#ffffff", ec="#43a047", lw=1.2))
    ax.text(135.5, 57.2, r"$\mathbf{u}_m^* = \arg\min_{\mathbf{u} \in \Delta_m} \mathbf{u}^T \mathbf{\Sigma}_m^- \mathbf{u}$", 
            ha="center", va="center", fontsize=8.0, fontweight="bold", color="#1b5e20")
    
    desc_stg1 = (
        "• Ward Linkage Dendrogram on D_ij^GNN\n"
        "• Bawa-Lindenberg Semi-Covariance Matrix\n"
        "• Intra-Cluster Weights Matrix: W_intra"
    )
    ax.text(122, 48.5, desc_stg1, ha="left", va="center", fontsize=7.0, color="#1b5e20")

    # Stage 2: Inter-Cluster
    box_stg2 = FancyBboxPatch((120, 7), 31, 32, boxstyle="round,pad=0.8",
                              fc=c_nco_bg, ec=c_nco_edge, lw=1.8)
    ax.add_patch(box_stg2)
    ax.text(135.5, 34.5, "Stage 2: Inter-Cluster\nRisk-Return Allocation", ha="center", va="center", **font_title)
    
    ax.add_patch(FancyBboxPatch((122, 23.5), 27, 7.5, boxstyle="round,pad=0.3", fc="#ffffff", ec="#2e7d32", lw=1.4))
    ax.text(135.5, 28.0, r"$\mathbf{v}^* = \arg\min_{\mathbf{v} \in \Delta_M} \left\{ \frac{1}{2}\mathbf{v}^T \mathbf{\Sigma}_{\mathrm{cl}} \mathbf{v} - \gamma \mathbf{v}^T \boldsymbol{\mu}_{\mathrm{cl}} \right\}$", 
            ha="center", va="center", fontsize=7.2, fontweight="bold", color="#1b5e20")
    ax.text(135.5, 24.8, r"$\mathbf{\Sigma}_{\mathrm{cl}} = \mathbf{W}_{\mathrm{intra}}^T \mathbf{\Sigma}^- \mathbf{W}_{\mathrm{intra}}$", 
            ha="center", va="center", fontsize=7.0, color="#33691e")

    desc_stg2 = (
        "• Cluster Bounds: 0.10 <= v_m <= 0.60\n"
        "• Single-Asset Diversification Cap: w_i <= 25%\n"
        "• Alternative Asset Cap: w_{BTC} <= 8%\n"
        "• Raw Composite Weight:\n"
        "   w_{i,t}^{raw} = v_{c(i)}^* * u_{c(i),i}^*"
    )
    ax.text(122, 13.5, desc_stg2, ha="left", va="center", fontsize=6.8, color="#1b5e20")

    # -------------------------------------------------------------
    # Phase IV: Execution & Turnover Control
    # -------------------------------------------------------------
    box_out = FancyBboxPatch((86, 7), 28, 25, boxstyle="round,pad=0.8",
                             fc=c_out_bg, ec=c_out_edge, lw=1.8)
    ax.add_patch(box_out)
    ax.text(100, 28.5, "Convex Turnover\nRegularization", ha="center", va="center", **font_title)
    
    ax.add_patch(FancyBboxPatch((88, 16.5), 24, 7.0, boxstyle="round,pad=0.3", fc="#ffffff", ec="#d84315", lw=1.4))
    ax.text(100, 21.0, r"$\mathbf{w}_t^* = (1-\lambda)\mathbf{w}_t^{\mathrm{raw}} + \lambda \mathbf{w}_{t-1}^*$", 
            ha="center", va="center", fontsize=8.0, fontweight="bold", color="#bf360c")
    ax.text(100, 18.0, r"Inertia Factor: $\lambda_{\mathrm{turn}} = 0.35$", ha="center", va="center", fontsize=7.2, color="#bf360c")

    desc_out = (
        "• Penalizes High Turnover Drag\n"
        "• Flat 10 bps Transaction Cost Model\n"
        "• Output: Rebalanced Portfolio Weights"
    )
    ax.text(88, 11.2, desc_out, ha="left", va="center", fontsize=6.8, color="#bf360c")

    # -------------------------------------------------------------
    # Connecting Arrows
    # -------------------------------------------------------------
    astyle = "Simple,tail_width=2.0,head_width=6.0,head_length=7"

    # Input -> TCN
    ax.add_patch(patches.FancyArrowPatch((22, 56), (25, 56), arrowstyle=astyle, color="#0288d1"))
    
    # TCN -> GAT
    ax.add_patch(patches.FancyArrowPatch((44, 56), (47, 56), arrowstyle=astyle, color="#0277bd"))
    
    # Topology -> GAT
    ax.add_patch(patches.FancyArrowPatch((44, 22.5), (47, 22.5), arrowstyle=astyle, color="#7b1fa2"))

    # GAT -> Metric Space
    ax.add_patch(patches.FancyArrowPatch((80, 56), (86, 56), arrowstyle=astyle, color="#4527a0"))

    # Metric Space -> Stage 1 NCO
    ax.add_patch(patches.FancyArrowPatch((114, 56), (120, 56), arrowstyle=astyle, color="#f57c00"))

    # Stage 1 -> Stage 2 NCO
    ax.add_patch(patches.FancyArrowPatch((135.5, 44), (135.5, 39), arrowstyle=astyle, color="#2e7d32"))

    # Stage 2 -> Phase IV Regularization
    ax.add_patch(patches.FancyArrowPatch((120, 20), (114, 20), arrowstyle=astyle, color="#d84315"))

    plt.tight_layout()
    pdf_path = os.path.join(FIGURES_DIR, "fig0_architecture.pdf")
    png_path = os.path.join(FIGURES_DIR, "fig0_architecture.png")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Architecture diagram generated:\n  PDF: {pdf_path}\n  PNG: {png_path}")

if __name__ == "__main__":
    draw_architecture()
