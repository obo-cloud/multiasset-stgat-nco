import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
from sklearn.metrics import silhouette_score

def compute_downside_semicovariance(returns, target_return=0.0):
    T, N = returns.shape
    downside_diff = np.minimum(returns - target_return, 0.0)
    semi_cov = np.dot(downside_diff.T, downside_diff) / float(T)
    sym_semi_cov = 0.5 * (semi_cov + semi_cov.T)
    eigvals = np.linalg.eigvalsh(sym_semi_cov)
    min_eig = np.min(eigvals)
    if min_eig < 1e-6:
        sym_semi_cov += (1e-6 - min_eig) * np.eye(N)
    return sym_semi_cov

def solve_constrained_min_variance(cov, max_weight=0.25, btc_idx=12, max_btc=0.10):
    N = cov.shape[0]
    w0 = np.ones(N) / N
    bounds = []
    for i in range(N):
        if i == btc_idx:
            bounds.append((0.0, max_btc))
        else:
            bounds.append((0.0, max_weight))
            
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    
    def obj(w):
        return np.dot(w.T, np.dot(cov, w))
        
    res = minimize(obj, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9, 'maxiter': 500})
    if res.success:
        w = np.maximum(res.x, 0.0)
        return w / np.sum(w)
    return w0

def solve_constrained_sortino(cov, mean_ret, rf=0.0, gamma=0.3, max_weight=0.25, btc_idx=12, max_btc=0.10):
    N = cov.shape[0]
    w0 = np.ones(N) / N
    bounds = []
    for i in range(N):
        if i == btc_idx:
            bounds.append((0.0, max_btc))
        else:
            bounds.append((0.0, max_weight))
            
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    
    def obj(w):
        var = np.dot(w.T, np.dot(cov, w))
        ret = np.dot(w, mean_ret) - rf
        return 0.5 * var - gamma * ret
        
    res = minimize(obj, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9, 'maxiter': 500})
    if res.success:
        w = np.maximum(res.x, 0.0)
        return w / np.sum(w)
    return solve_constrained_min_variance(cov, max_weight=max_weight, btc_idx=btc_idx, max_btc=max_btc)

def cluster_from_distance_matrix(dist_matrix, min_clusters=2, max_clusters=4):
    N = dist_matrix.shape[0]
    condensed_dist = squareform(dist_matrix, checks=False)
    link = sch.linkage(condensed_dist, method='ward')
    
    best_k = min_clusters
    best_score = -1.0
    for k in range(min_clusters, min(max_clusters + 1, N)):
        labels = sch.fcluster(link, k, criterion='maxclust')
        if len(np.unique(labels)) > 1:
            score = silhouette_score(dist_matrix, labels, metric='precomputed')
            if score > best_score:
                best_score = score
                best_k = k
    clusters = sch.fcluster(link, best_k, criterion='maxclust')
    return clusters, best_k

def allocate_nco_constrained(cov, mean_ret, dist_matrix, gamma=0.3, max_weight=0.25, btc_idx=12, max_btc=0.10):
    N = cov.shape[0]
    clusters, num_clusters = cluster_from_distance_matrix(dist_matrix)
    unique_clusters = np.unique(clusters)
    
    # 1. Intra-cluster weights
    w_intra = np.zeros((N, len(unique_clusters)))
    for idx, c_id in enumerate(unique_clusters):
        c_mask = (clusters == c_id)
        sub_cov = cov[np.ix_(c_mask, c_mask)]
        sub_N = np.sum(c_mask)
        
        # Intra-cluster min-variance with bounds
        w0_sub = np.ones(sub_N) / sub_N
        sub_bounds = [(0.0, 1.0) for _ in range(sub_N)]
        cons_sub = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        res_sub = minimize(lambda w: np.dot(w.T, np.dot(sub_cov, w)), w0_sub, 
                           method='SLSQP', bounds=sub_bounds, constraints=cons_sub)
        if res_sub.success:
            w_intra[c_mask, idx] = np.maximum(res_sub.x, 0.0)
        else:
            w_intra[c_mask, idx] = w0_sub
            
    # 2. Inter-cluster allocation
    cl_cov = np.dot(w_intra.T, np.dot(cov, w_intra))
    cl_mean = np.dot(w_intra.T, mean_ret)
    
    # Inter-cluster: Inverse Variance / Minimum Variance across clusters
    M = len(unique_clusters)
    w0_cl = np.ones(M) / M
    cl_bounds = [(0.10, 0.60) for _ in range(M)] # Ensure no single cluster is starved or dominates > 60%
    cons_cl = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    
    def obj_cl(w):
        var = np.dot(w.T, np.dot(cl_cov, w))
        ret = np.dot(w, cl_mean)
        return 0.5 * var - gamma * ret
        
    res_cl = minimize(obj_cl, w0_cl, method='SLSQP', bounds=cl_bounds, constraints=cons_cl)
    if res_cl.success:
        w_inter = np.maximum(res_cl.x, 0.0)
        w_inter /= np.sum(w_inter)
    else:
        w_inter = w0_cl
        
    # 3. Composite weights
    w_comp = np.dot(w_intra, w_inter)
    
    # Enforce asset-level max caps
    w_comp = np.minimum(w_comp, max_weight)
    if btc_idx is not None and btc_idx < N:
        w_comp[btc_idx] = min(w_comp[btc_idx], max_btc)
    w_comp = np.maximum(w_comp, 0.0)
    return w_comp / np.sum(w_comp)

def allocate_hrp_constrained(cov, max_weight=0.25, btc_idx=12, max_btc=0.10):
    corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(0.5 * (1.0 - corr))
    np.fill_diagonal(dist, 0.0)
    
    link = sch.linkage(squareform(dist, checks=False), method='single')
    sort_ix = sch.leaves_list(link)
    
    w = np.ones(len(sort_ix))
    cluster_items = [sort_ix]
    
    while len(cluster_items) > 0:
        next_items = []
        for items in cluster_items:
            if len(items) > 1:
                mid = len(items) // 2
                left = items[:mid]
                right = items[mid:]
                
                cov_l = cov[np.ix_(left, left)]
                cov_r = cov[np.ix_(right, right)]
                
                ivp_l = 1.0 / np.diag(cov_l)
                w_l = ivp_l / np.sum(ivp_l)
                ivp_r = 1.0 / np.diag(cov_r)
                w_r = ivp_r / np.sum(ivp_r)
                
                var_l = np.dot(w_l.T, np.dot(cov_l, w_l))
                var_r = np.dot(w_r.T, np.dot(cov_r, w_r))
                
                alpha = 1.0 - var_l / (var_l + var_r)
                w[left] *= alpha
                w[right] *= (1.0 - alpha)
                
                next_items.append(left)
                next_items.append(right)
        cluster_items = next_items
        
    w = np.minimum(w, max_weight)
    if btc_idx is not None and btc_idx < len(w):
        w[btc_idx] = min(w[btc_idx], max_btc)
    w = np.maximum(w, 0.0)
    return w / np.sum(w)
