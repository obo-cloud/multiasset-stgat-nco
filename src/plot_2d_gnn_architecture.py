import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

def draw_2d_gnn_architecture():
    # 2D Flat Academic Layout - 16 x 8.8 inches
    fig, ax = plt.subplots(figsize=(16, 8.8), dpi=300)
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 88)
    ax.axis("off")

    # Clean 2D Color Palette (Flat Design)
    c_p1_bg = "#f0f7fa"
    c_p1_border = "#0288d1"
    
    c_p2_bg = "#f4f1fa"
    c_p2_border = "#5e35b1"
    
    c_p3_bg = "#fffbf0"
    c_p3_border = "#f57c00"
    
    c_p4_bg = "#f1f8e9"
    c_p4_border = "#388e3c"

    # Font styles
    f_header = {"fontsize": 10.5, "fontweight": "bold"}
    f_title = {"fontsize": 9.5, "fontweight": "bold", "color": "#1a252f"}
    f_sub = {"fontsize": 8.0, "fontweight": "bold", "color": "#2c3e50"}
    f_desc = {"fontsize": 7.2, "color": "#34495e"}

    # =========================================================================
    # 4 MAIN 2D PANELS
    # =========================================================================
    
    # -------------------------------------------------------------------------
    # PANEL 1: Temporal Feature Extraction (x: 2 to 38, y: 3 to 84)
    # -------------------------------------------------------------------------
    box_p1 = FancyBboxPatch((2, 3), 36, 81, boxstyle="round,pad=1.2",
                            fc=c_p1_bg, ec=c_p1_border, lw=1.5, ls="--")
    ax.add_patch(box_p1)
    ax.text(20, 81.5, "1. TEMPORAL FEATURE EXTRACTION", ha="center", va="center", 
            color="#01579b", **f_header)

    # 1A. Multi-Asset Input Tensor
    b_in = FancyBboxPatch((4, 46), 32, 31, boxstyle="round,pad=0.8", fc="#ffffff", ec="#03a9f4", lw=1.4)
    ax.add_patch(b_in)
    ax.text(20, 74, "Multi-Asset Temporal Tensor", ha="center", va="center", **f_title)
    ax.text(20, 68.5, r"$\mathbf{X}_t \in \mathbb{R}^{N \times F \times K}$", ha="center", va="center", 
            fontsize=10.0, fontweight="bold", color="#0277bd")
    
    in_items = [
        r"$\bullet$ $N = 13$ Multi-Asset Classes",
        r"$\bullet$ $K = 12$ Rolling Month Horizon",
        r"$\bullet$ $F = 5$ Dynamic Node Features:",
        r"   $f_1$: Realized Monthly Simple Return ($r_{i,t}$)",
        r"   $f_2$: Realized 3-Month Volatility ($\sigma_{i,t}$)",
        r"   $f_3$: Downside Semi-Deviation ($\sigma_{i,t}^-$)",
        r"   $f_4$: 3M/12M Cumulative Momentum ($m_{i,t}$)",
        r"   $f_5$: Relative Market Strength ($\beta_{i,t}$)"
    ]
    for idx, item in enumerate(in_items):
        ax.text(5.5, 63.5 - idx * 2.2, item, ha="left", va="center", fontsize=7.1, color="#263238")

    # 1B. Causal Dilated 1D TCN
    b_tcn = FancyBboxPatch((4, 8), 32, 33, boxstyle="round,pad=0.8", fc="#ffffff", ec="#0288d1", lw=1.4)
    ax.add_patch(b_tcn)
    ax.text(20, 37.5, "Causal Dilated 1D-CNN (TCN)", ha="center", va="center", **f_title)
    ax.text(20, 32.5, r"$\mathbf{h}_i^{(0)} = \mathrm{TemporalConv}(\mathbf{X}_{i,t}) \in \mathbb{R}^{d_{\mathrm{temp}}}$", 
            ha="center", va="center", fontsize=8.2, fontweight="bold", color="#01579b")
    
    tcn_items = [
        r"$\bullet$ Dilated Causal 1D Convolution Kernels",
        r"$\bullet$ Zero-Padding to Guarantee Strict Causality",
        r"$\bullet$ Non-Linear LeakyReLU Activation Path",
        r"$\bullet$ Residual Identity Highway Skip Connection",
        r"$\bullet$ Encodes Asset-Specific Momentum Memory",
        r"$\bullet$ Insulates Time Series from Lookahead Bias"
    ]
    for idx, item in enumerate(tcn_items):
        ax.text(5.5, 27.5 - idx * 2.5, item, ha="left", va="center", fontsize=7.2, color="#263238")

    # Arrow 1A -> 1B
    ax.add_patch(FancyArrowPatch((20, 46), (20, 41), arrowstyle="Simple,tail_width=2.0,head_width=6,head_length=7", 
                                 color="#0288d1"))

    # -------------------------------------------------------------------------
    # PANEL 2: INTERCONNECTED DYNAMIC GNN & MULTI-HEAD GAT (x: 41 to 93, y: 3 to 84)
    # (USER HIGHLIGHT: "gambar gnn yang saling terhubung 2D")
    # -------------------------------------------------------------------------
    box_p2 = FancyBboxPatch((41, 3), 52, 81, boxstyle="round,pad=1.2",
                            fc=c_p2_bg, ec=c_p2_border, lw=1.6, ls="--")
    ax.add_patch(box_p2)
    ax.text(67, 81.5, "2. INTERCONNECTED GRAPH ATTENTION NETWORK (GNN / GAT)", 
            ha="center", va="center", color="#4527a0", **f_header)

    # 2A. Real Interconnected Multi-Asset Graph Topology
    b_graph = FancyBboxPatch((43, 44), 48, 33, boxstyle="round,pad=0.8", fc="#ffffff", ec="#7e57c2", lw=1.5)
    ax.add_patch(b_graph)
    ax.text(67, 74.5, "Interconnected Multi-Asset Graph", ha="center", va="center", **f_title)
    ax.text(67, 70.8, r"$\mathcal{G}_t = (\mathcal{V}, \mathcal{E}_t, \mathbf{A}_t), \quad A_{ij,t} = \mathbf{1}_{\{|C_{ij,t}| \geq \tau \ \mathrm{or} \ i=j\}}$", 
            ha="center", va="center", fontsize=7.8, fontweight="bold", color="#512da8")

    # Coordinates for the 13 Interconnected Asset Nodes in 2D Space
    # Clustered meaningfully by asset class in 2D
    nodes = {
        # Equities (Blue-Indigo)
        "SPY": (49, 61), "QQQ": (55, 65), "EEM": (50, 52),
        # Treasuries & Fixed Income (Emerald-Teal)
        "IEF": (64, 62), "TLT": (71, 65), "HYG": (62, 53), "LQD": (70, 54),
        # Real Assets & Commodities (Amber-Orange)
        "GLD": (79, 63), "DBC": (85, 59), "VNQ": (78, 52),
        # Cash & Currencies (Slate-Brown)
        "UUP": (86, 66), "SHV": (86, 49),
        # Digital Assets (Purple-Magenta)
        "BTC": (57, 47)
    }

    node_colors = {
        "SPY": "#1976d2", "QQQ": "#1565c0", "EEM": "#0288d1",
        "IEF": "#388e3c", "TLT": "#2e7d32", "HYG": "#43a047", "LQD": "#1b5e20",
        "GLD": "#f57c00", "DBC": "#ef6c00", "VNQ": "#e65100",
        "UUP": "#5d4037", "SHV": "#455a64",
        "BTC": "#8e24aa"
    }

    # Interconnected Edges across the Multi-Asset Graph (Dynamic Correlation Network)
    inter_edges = [
        # Equity cluster
        ("SPY", "QQQ", 2.2, "#3f51b5", "-"),
        ("SPY", "EEM", 1.4, "#3f51b5", "-"),
        ("QQQ", "EEM", 1.2, "#3f51b5", "-"),
        # Fixed income cluster
        ("IEF", "TLT", 2.4, "#2e7d32", "-"),
        ("IEF", "LQD", 1.8, "#2e7d32", "-"),
        ("TLT", "LQD", 1.5, "#2e7d32", "-"),
        ("HYG", "LQD", 2.0, "#388e3c", "-"),
        ("HYG", "SPY", 1.6, "#5c6bc0", "--"), # Credit-Equity contagion link
        # Real assets cluster
        ("GLD", "DBC", 1.4, "#e65100", "-"),
        ("GLD", "IEF", 1.5, "#00897b", "-"), # Flight to safety
        ("DBC", "VNQ", 1.2, "#e65100", "-"),
        ("VNQ", "SPY", 1.5, "#5c6bc0", "--"),
        # Cash & Currency
        ("UUP", "GLD", 1.6, "#d84315", ":"), # USD vs Gold inverse/hedge
        ("SHV", "IEF", 1.8, "#2e7d32", "-"),
        ("UUP", "TLT", 1.3, "#455a64", "--"),
        # Bitcoin risk linkage
        ("BTC", "QQQ", 1.7, "#8e24aa", "--"), # High-beta tech link
        ("BTC", "SPY", 1.4, "#8e24aa", "--"),
        ("BTC", "GLD", 1.2, "#8e24aa", ":")
    ]

    # Draw Interconnected Edges
    for n1, n2, lw, col, ls in inter_edges:
        p1 = nodes[n1]
        p2 = nodes[n2]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=col, lw=lw, ls=ls, alpha=0.65, zorder=2)

    # Draw Interconnected Nodes
    for name, (x, y) in nodes.items():
        circ = Circle((x, y), 2.2, fc="#ffffff", ec=node_colors[name], lw=1.8, zorder=4)
        ax.add_patch(circ)
        ax.text(x, y, name, ha="center", va="center", fontsize=6.8, fontweight="bold", 
                color=node_colors[name], zorder=5)

    ax.text(67, 45.2, "Interconnected cross-asset graph with dynamic contagion & safety-flight edges", 
            ha="center", va="center", fontsize=6.5, fontstyle="italic", color="#546e7a")

    # 2B. Detailed GAT Multi-Head Attention Message Passing Mechanism
    b_gat = FancyBboxPatch((43, 6), 48, 35, boxstyle="round,pad=0.8", fc="#ffffff", ec="#5e35b1", lw=1.5)
    ax.add_patch(b_gat)
    ax.text(67, 38.0, "Multi-Head Attention Message Passing", ha="center", va="center", **f_title)

    # Formula 1: Attention Coefficients
    ax.add_patch(FancyBboxPatch((45, 27.5), 44, 7.8, boxstyle="round,pad=0.4", fc="#ede7f6", ec="#7e57c2", lw=1.2))
    ax.text(67, 32.8, r"$\alpha_{ij}^{(k)} = \frac{\exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}_k^T [\mathbf{W}^{(k)}\mathbf{h}_i \,\|\, \mathbf{W}^{(k)}\mathbf{h}_j]\right)\right)}"
                      r"{\sum_{l \in \mathcal{N}_i} \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}_k^T [\mathbf{W}^{(k)}\mathbf{h}_i \,\|\, \mathbf{W}^{(k)}\mathbf{h}_l]\right)\right)}$",
            ha="center", va="center", fontsize=7.2, fontweight="bold", color="#311b92")
    ax.text(67, 28.8, "Self-Attention Weight over Edge (i, j) under Head k", ha="center", va="center", 
            fontsize=6.5, color="#512da8")

    # Formula 2: Multi-Head Aggregation
    ax.add_patch(FancyBboxPatch((45, 17.5), 44, 8.5, boxstyle="round,pad=0.4", fc="#ede7f6", ec="#512da8", lw=1.3))
    ax.text(67, 23.0, r"$\mathbf{z}_i = \sigma \left( \sum_{k=1}^{K_h} \sum_{j \in \mathcal{N}_i} \alpha_{ij}^{(k)} \mathbf{W}^{(k)} \mathbf{h}_j \right)$", 
            ha="center", va="center", fontsize=8.0, fontweight="bold", color="#1a237e")
    ax.text(67, 19.0, "Aggregated Spatial Node Representation in Latent Space", ha="center", va="center", 
            fontsize=6.5, color="#311b92")

    gat_bullets = [
        r"$\bullet$ Multi-Head Attention: $K_h = 2$ heads to decouple complementary risk regimes",
        r"$\bullet$ Graph Laplacian Smoothing Regularization preserves manifold stability",
        r"$\bullet$ Captures High-Order Multi-Hop Contagion without correlation saturation"
    ]
    for idx, b in enumerate(gat_bullets):
        ax.text(45.5, 14.5 - idx * 2.3, b, ha="left", va="center", fontsize=6.8, color="#263238")

    # Arrow 1 -> 2
    ax.add_patch(FancyArrowPatch((38, 59), (43, 59), arrowstyle="Simple,tail_width=2.5,head_width=7,head_length=8", 
                                 color="#5e35b1"))
    # Arrow 2A -> 2B
    ax.add_patch(FancyArrowPatch((67, 44), (67, 41), arrowstyle="Simple,tail_width=2.0,head_width=6,head_length=7", 
                                 color="#5e35b1"))

    # -------------------------------------------------------------------------
    # PANEL 3: METRIC GEOMETRY & CLUSTERING (x: 96 to 126, y: 3 to 84)
    # -------------------------------------------------------------------------
    box_p3 = FancyBboxPatch((96, 3), 30, 81, boxstyle="round,pad=1.2",
                            fc=c_p3_bg, ec=c_p3_border, lw=1.5, ls="--")
    ax.add_patch(box_p3)
    ax.text(111, 81.5, "3. METRIC GEOMETRY & TREE", ha="center", va="center", 
            color="#e65100", **f_header)

    # 3A. Hypersphere Normalization & Distance
    b_dist = FancyBboxPatch((98, 46), 26, 31, boxstyle="round,pad=0.8", fc="#ffffff", ec="#ff9800", lw=1.4)
    ax.add_patch(b_dist)
    ax.text(111, 74, "Hypersphere Distance", ha="center", va="center", **f_title)
    
    ax.add_patch(FancyBboxPatch((100, 64), 22, 6.2, boxstyle="round,pad=0.3", fc="#fff8e1", ec="#ffa000", lw=1.2))
    ax.text(111, 67.1, r"$\tilde{\mathbf{z}}_i = \frac{\mathbf{z}_i}{\|\mathbf{z}_i\|_2} \in \mathbb{S}^{d-1}$", 
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="#e65100")
    
    ax.add_patch(FancyBboxPatch((100, 54), 22, 7.5, boxstyle="round,pad=0.3", fc="#fff8e1", ec="#e65100", lw=1.3))
    ax.text(111, 58.8, r"$D_{ij}^{\mathrm{GNN}} = \sqrt{\frac{1}{2}(1 - \tilde{\mathbf{z}}_i^T \tilde{\mathbf{z}}_j)}$", 
            ha="center", va="center", fontsize=8.0, fontweight="bold", color="#b71c1c")
    ax.text(111, 55.5, r"$D_{ij}^{\mathrm{GNN}} \in [0, 1]$ (Angular Metric)", ha="center", va="center", 
            fontsize=6.5, color="#bf360c")

    dist_desc = [
        r"$\bullet$ Non-Euclidean Angular Metric",
        r"$\bullet$ Bounded, Positive Definite",
        r"$\bullet$ True Metric Space Satisfying",
        r"  Triangle Inequality"
    ]
    for idx, d in enumerate(dist_desc):
        ax.text(100, 50.5 - idx * 2.3, d, ha="left", va="center", fontsize=6.8, color="#37474f")

    # 3B. Hierarchical Dendrogram
    b_dend = FancyBboxPatch((98, 8), 26, 33, boxstyle="round,pad=0.8", fc="#ffffff", ec="#f57c00", lw=1.4)
    ax.add_patch(b_dend)
    ax.text(111, 37.5, "Ward Tree Linkage", ha="center", va="center", **f_title)

    # Mini 2D Dendrogram illustration
    # Draw simple dendrogram tree
    ax.plot([101, 105], [30, 30], color="#d84315", lw=1.4)
    ax.plot([101, 101], [27, 30], color="#d84315", lw=1.4)
    ax.plot([105, 105], [27, 30], color="#d84315", lw=1.4)
    ax.text(101, 25.5, "Equities", ha="center", va="center", fontsize=5.8, color="#1565c0")
    ax.text(105, 25.5, "Crypto", ha="center", va="center", fontsize=5.8, color="#8e24aa")

    ax.plot([113, 117], [30, 30], color="#2e7d32", lw=1.4)
    ax.plot([113, 113], [27, 30], color="#2e7d32", lw=1.4)
    ax.plot([117, 117], [27, 30], color="#2e7d32", lw=1.4)
    ax.text(113, 25.5, "Bonds", ha="center", va="center", fontsize=5.8, color="#2e7d32")
    ax.text(117, 25.5, "Cash", ha="center", va="center", fontsize=5.8, color="#455a64")

    # Higher root
    ax.plot([103, 115], [33, 33], color="#4527a0", lw=1.4)
    ax.plot([103, 103], [30, 33], color="#4527a0", lw=1.4)
    ax.plot([115, 115], [30, 33], color="#4527a0", lw=1.4)

    dend_items = [
        r"$\bullet$ Ward Minimum-Variance Linkage",
        r"$\bullet$ Silhouette Validated: $M \in [2, 4]$",
        r"$\bullet$ Disjoint Clusters: $\mathcal{I}_1, \dots, \mathcal{I}_M$",
        r"$\bullet$ Isolates Orthogonal Risk Blocks"
    ]
    for idx, it in enumerate(dend_items):
        ax.text(100, 20.0 - idx * 2.5, it, ha="left", va="center", fontsize=6.8, color="#37474f")

    # Arrow 2 -> 3
    ax.add_patch(FancyArrowPatch((91, 59), (96, 59), arrowstyle="Simple,tail_width=2.5,head_width=7,head_length=8", 
                                 color="#f57c00"))
    # Arrow 3A -> 3B
    ax.add_patch(FancyArrowPatch((111, 46), (111, 41), arrowstyle="Simple,tail_width=2.0,head_width=6,head_length=7", 
                                 color="#f57c00"))

    # -------------------------------------------------------------------------
    # PANEL 4: TWO-STAGE DOWNSIDE NCO & EXECUTION (x: 129 to 158, y: 3 to 84)
    # -------------------------------------------------------------------------
    box_p4 = FancyBboxPatch((129, 3), 29, 81, boxstyle="round,pad=1.2",
                            fc=c_p4_bg, ec=c_p4_border, lw=1.5, ls="--")
    ax.add_patch(box_p4)
    ax.text(143.5, 81.5, "4. DOWNSIDE NCO & EXECUTION", ha="center", va="center", 
            color="#1b5e20", **f_header)

    # 4A. Stage 1: Intra-Cluster Downside Min-Variance
    b_stg1 = FancyBboxPatch((131, 53), 25, 24, boxstyle="round,pad=0.8", fc="#ffffff", ec="#43a047", lw=1.4)
    ax.add_patch(b_stg1)
    ax.text(143.5, 74, "Stage 1: Intra-Cluster", ha="center", va="center", **f_title)
    
    ax.add_patch(FancyBboxPatch((132.5, 64.5), 22, 6.5, boxstyle="round,pad=0.3", fc="#e8f5e9", ec="#2e7d32", lw=1.2))
    ax.text(143.5, 67.8, r"$\mathbf{u}_m^* = \arg\min_{\mathbf{u} \in \Delta_m} \mathbf{u}^T \mathbf{\Sigma}_m^- \mathbf{u}$", 
            ha="center", va="center", fontsize=7.2, fontweight="bold", color="#1b5e20")

    stg1_desc = [
        r"$\bullet$ Bawa-Lindenberg Semi-Cov:",
        r"  $\Sigma_{ij}^- = \frac{1}{L} \sum \min(r_i,0)\min(r_j,0)$",
        r"$\bullet$ Local Intra-Cluster Matrix $\mathbf{W}_{\mathrm{intra}}$",
        r"$\bullet$ Quenches Intra-Block Noise"
    ]
    for idx, s in enumerate(stg1_desc):
        ax.text(132, 61.5 - idx * 2.3, s, ha="left", va="center", fontsize=6.5, color="#1b5e20")

    # 4B. Stage 2: Inter-Cluster Downside Allocation
    b_stg2 = FancyBboxPatch((131, 28), 25, 22, boxstyle="round,pad=0.8", fc="#ffffff", ec="#2e7d32", lw=1.4)
    ax.add_patch(b_stg2)
    ax.text(143.5, 47, "Stage 2: Inter-Cluster", ha="center", va="center", **f_title)
    
    ax.add_patch(FancyBboxPatch((132.5, 37.5), 22, 6.8, boxstyle="round,pad=0.3", fc="#e8f5e9", ec="#1b5e20", lw=1.3))
    ax.text(143.5, 41.5, r"$\mathbf{v}^* = \arg\min_{\mathbf{v} \in \Delta_M} \left\{ \frac{1}{2} \mathbf{v}^T \mathbf{\Sigma}_{\mathrm{cl}} \mathbf{v} - \gamma \mathbf{v}^T \mathbf{\mu}_{\mathrm{cl}} \right\}$", 
            ha="center", va="center", fontsize=6.8, fontweight="bold", color="#1b5e20")
    ax.text(143.5, 38.5, r"$\mathbf{\Sigma}_{\mathrm{cl}} = \mathbf{W}_{\mathrm{intra}}^T \mathbf{\Sigma}^- \mathbf{W}_{\mathrm{intra}}$", 
            ha="center", va="center", fontsize=6.5, color="#2e7d32")

    stg2_desc = [
        r"$\bullet$ Cluster Bounds: $0.10 \leq v_m \leq 0.60$",
        r"$\bullet$ Raw: $w_{i,t}^{\mathrm{raw}} = v_{c(i)}^* \cdot u_{c(i),i}^*$",
        r"$\bullet$ Caps: $w_i \leq 25\%$, $w_{\mathrm{BTC}} \leq 8\%$"
    ]
    for idx, s in enumerate(stg2_desc):
        ax.text(132, 35.0 - idx * 2.3, s, ha="left", va="center", fontsize=6.5, color="#1b5e20")

    # 4C. Tactical Rebalancing & Turnover Regularization
    b_rebal = FancyBboxPatch((131, 7), 25, 18, boxstyle="round,pad=0.8", fc="#ffffff", ec="#d84315", lw=1.4)
    ax.add_patch(b_rebal)
    ax.text(143.5, 22.5, "Tactical Rebalancing", ha="center", va="center", **f_title)
    
    ax.add_patch(FancyBboxPatch((132.5, 13.5), 22, 6.2, boxstyle="round,pad=0.3", fc="#fbe9e7", ec="#bf360c", lw=1.3))
    ax.text(143.5, 17.2, r"$\mathbf{w}_t^* = (1-\lambda)\mathbf{w}_t^{\mathrm{raw}} + \lambda \mathbf{w}_{t-1}^*$", 
            ha="center", va="center", fontsize=7.2, fontweight="bold", color="#bf360c")
    ax.text(143.5, 14.5, r"Turnover Inertia: $\lambda_{\mathrm{turn}} = 0.35$", ha="center", va="center", 
            fontsize=6.5, color="#d84315")
    
    ax.text(132, 10.5, r"$\bullet$ 10 bps Friction Net Accounting", ha="left", va="center", fontsize=6.5, color="#bf360c")
    ax.text(132, 8.5, r"$\bullet$ Rebalanced Weights Output $\mathbf{w}_t^*$", ha="left", va="center", fontsize=6.5, color="#bf360c")

    # Arrows in Panel 4
    # 3 -> 4
    ax.add_patch(FancyArrowPatch((124, 60), (131, 60), arrowstyle="Simple,tail_width=2.5,head_width=7,head_length=8", 
                                 color="#388e3c"))
    # 4A -> 4B
    ax.add_patch(FancyArrowPatch((143.5, 53), (143.5, 50), arrowstyle="Simple,tail_width=2.0,head_width=6,head_length=7", 
                                 color="#2e7d32"))
    # 4B -> 4C
    ax.add_patch(FancyArrowPatch((143.5, 28), (143.5, 25), arrowstyle="Simple,tail_width=2.0,head_width=6,head_length=7", 
                                 color="#d84315"))

    plt.tight_layout()
    pdf_path = os.path.join(FIGURES_DIR, "fig1_gnn_architecture_2d.pdf")
    png_path = os.path.join(FIGURES_DIR, "fig1_gnn_architecture_2d.png")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved 2D GNN Architecture:\n  PDF: {pdf_path}\n  PNG: {png_path}")

if __name__ == "__main__":
    draw_2d_gnn_architecture()
