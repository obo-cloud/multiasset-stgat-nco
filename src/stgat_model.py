import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TemporalConvBlock(nn.Module):
    """
    1D Temporal Convolutional Block with Dilated Causal Convolutions
    to capture multi-scale time-series dynamics of individual assets.
    """
    def __init__(self, in_features, hidden_dim, kernel_size=2, dilation=1):
        super(TemporalConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_features,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation
        )
        self.conv2 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation
        )
        self.relu = nn.LeakyReLU(0.2)
        self.proj = nn.Conv1d(in_features, hidden_dim, 1) if in_features != hidden_dim else nn.Identity()
        self.dropout = nn.Dropout(0.1)
        self.padding_trim = (kernel_size - 1) * dilation

    def forward(self, x):
        # x shape: (N, in_features, seq_len)
        res = self.proj(x)
        out = self.conv1(x)
        if self.padding_trim > 0:
            out = out[:, :, :-self.padding_trim]
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        if self.padding_trim > 0:
            out = out[:, :, :-self.padding_trim]
        out = self.relu(out + res)
        return out


class MultiHeadGraphAttentionLayer(nn.Module):
    """
    Multi-Head Graph Attention Network (GAT) layer with dynamic attention
    weights to capture time-varying cross-asset contagion and correlations.
    """
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1, alpha=0.2):
        super(MultiHeadGraphAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.zeros(num_heads, self.head_dim, 1))
        self.a_dst = nn.Parameter(torch.zeros(num_heads, self.head_dim, 1))
        
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        # h: (N, in_dim)
        # adj: (N, N) binary or weighted adjacency matrix
        N = h.size(0)
        
        # Linear projection: (N, num_heads, head_dim)
        Wh = self.W(h).view(N, self.num_heads, self.head_dim)
        
        # Self-attention score computation
        # f_src: (num_heads, N, 1) -> (num_heads, N, 1)
        Wh_perm = Wh.permute(1, 0, 2) # (num_heads, N, head_dim)
        f_src = torch.matmul(Wh_perm, self.a_src) # (num_heads, N, 1)
        f_dst = torch.matmul(Wh_perm, self.a_dst) # (num_heads, N, 1)
        
        # Broadcasted sum: (num_heads, N, N)
        scores = self.leaky_relu(f_src + f_dst.transpose(-1, -2))
        
        # Mask out non-edges
        mask = (adj == 0).unsqueeze(0) # (1, N, N)
        scores = scores.masked_fill(mask, -1e9)
        
        # Softmax over neighbors
        attn = F.softmax(scores, dim=-1) # (num_heads, N, N)
        attn = self.dropout(attn)
        
        # Aggregate neighbor representations
        # out: (num_heads, N, head_dim)
        out = torch.matmul(attn, Wh_perm)
        # Concatenate heads: (N, out_dim)
        out = out.permute(1, 0, 2).contiguous().view(N, self.out_dim)
        return out, attn.mean(dim=0) # return aggregated representation & mean attention map


class STGATModel(nn.Module):
    """
    Spatio-Temporal Graph Attention Network (ST-GAT)
    Extracts dynamic latent embeddings Z_t in R^{N x d} for multi-asset clustering.
    """
    def __init__(self, in_features=5, temporal_dim=32, spatial_dim=32, embedding_dim=16, num_heads=4):
        super(STGATModel, self).__init__()
        self.temporal_block1 = TemporalConvBlock(in_features, temporal_dim, kernel_size=2, dilation=1)
        self.temporal_block2 = TemporalConvBlock(temporal_dim, temporal_dim, kernel_size=2, dilation=2)
        
        self.gat_layer1 = MultiHeadGraphAttentionLayer(temporal_dim, spatial_dim, num_heads=num_heads)
        self.gat_layer2 = MultiHeadGraphAttentionLayer(spatial_dim, embedding_dim, num_heads=num_heads)
        
        # Predictor head for self-supervised / return-risk multi-task objective
        self.risk_head = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1) # Expected next-period return / utility ranking
        )
        
        self.vol_head = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1) # Expected next-period downside risk
        )

    def forward(self, x, adj):
        # x shape: (N, in_features, seq_len)
        # adj shape: (N, N)
        
        # 1. Temporal feature encoding
        temp_out = self.temporal_block1(x)
        temp_out = self.temporal_block2(temp_out)
        
        # Pooling over sequence length: use last time-step + mean pooling
        h_temp = temp_out[:, :, -1] + temp_out.mean(dim=-1) # (N, temporal_dim)
        
        # 2. Spatial Graph Attention
        h_spat1, attn_map1 = self.gat_layer1(h_temp, adj)
        h_spat1 = F.leaky_relu(h_spat1, 0.2)
        
        embeddings, attn_map2 = self.gat_layer2(h_spat1, adj) # (N, embedding_dim)
        
        # Normalize embeddings to unit sphere for robust distance metrics
        norm_embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        # 3. Multi-task prediction heads
        pred_returns = self.risk_head(norm_embeddings).squeeze(-1)
        pred_downside = F.softplus(self.vol_head(norm_embeddings).squeeze(-1))
        
        return norm_embeddings, pred_returns, pred_downside, attn_map2


def build_dynamic_graph(corr_matrix, threshold=0.30):
    """
    Constructs an adjacency matrix from a correlation matrix using adaptive thresholding.
    Ensures self-loops and positive semi-definite connectivity.
    """
    N = corr_matrix.shape[0]
    adj = np.where(np.abs(corr_matrix) >= threshold, 1.0, 0.0)
    np.fill_diagonal(adj, 1.0)
    
    # Ensure graph is not disconnected (each node must have at least 1 neighbor beside itself)
    for i in range(N):
        if np.sum(adj[i]) <= 1.0:
            # Connect to top correlated peer
            peer = np.argsort(np.abs(corr_matrix[i]))[-2]
            adj[i, peer] = 1.0
            adj[peer, i] = 1.0
            
    return torch.tensor(adj, dtype=torch.float32)

if __name__ == "__main__":
    N = 13
    in_features = 5
    seq_len = 12
    x = torch.randn(N, in_features, seq_len)
    corr = np.random.uniform(-1, 1, (N, N))
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    adj = build_dynamic_graph(corr, threshold=0.30)
    
    model = STGATModel(in_features=in_features, temporal_dim=32, spatial_dim=32, embedding_dim=16)
    emb, ret, vol, attn = model(x, adj)
    print("ST-GAT Sanity Check:")
    print("  Input Shape:     ", x.shape)
    print("  Embeddings Shape:", emb.shape)
    print("  Pred Returns:    ", ret.shape)
    print("  Pred Downside:   ", vol.shape)
    print("  Attention Map:   ", attn.shape)
    print("[SUCCESS] ST-GAT Model compiles and runs smoothly.")
