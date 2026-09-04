import torch
import numpy as np


def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


# ----------------------------- 图构建 -----------------------------
def neighborhood(feat, k, metric='cosine', spec_ang=False):
    """Build a (directed) k-nearest-neighbour adjacency.

    Parameters
    ----------
    feat : np.ndarray, shape (n_features, n_nodes)
        Columns are nodes (samples), rows are their feature vectors.
    k : int
        Number of neighbours to keep for every node.
    metric : {'cosine', 'euclidean'}
        Distance used to rank neighbours.
    spec_ang : bool
        Only meaningful together with metric='euclidean'; converts the
        Euclidean gram matrix into a spectral-angle-like distance.

    Returns
    -------
    C : np.ndarray, shape (n_nodes, n_nodes)
        Binary k-NN matrix (row i -> its k nearest neighbours).
    """
    if metric == 'euclidean':
        # ----- 欧氏距离 -----
        featprod = np.dot(feat.T, feat)
        smat = np.tile(np.diag(featprod), (feat.shape[1], 1))
        if spec_ang:
            dmat = 1 - featprod / np.sqrt(smat * smat.T)  # 1 - spectral angle
        else:
            dmat = smat + smat.T - 2 * featprod
    else:  # ----- 余弦距离 -----
        norms = np.linalg.norm(feat, axis=0, keepdims=True)
        norms = np.where(norms == 0, 1e-12, norms)  # 避免除零
        cosine_similarity = np.dot(feat.T, feat) / (norms.T @ norms)
        dmat = 1 - cosine_similarity

    dsort = np.argsort(dmat)[:, 1:k + 1]
    C = np.zeros((feat.shape[1], feat.shape[1]))
    for i in range(feat.shape[1]):
        for j in dsort[i]:
            C[i, j] = 1.0

    return C


def norm_adj(feat, eud=False, k=2):
    """Symmetric-normalised adjacency of the k-NN sample graph.

    feat : array-like, shape (n_samples, n_features)
    eud   : if True use Euclidean distance, otherwise cosine distance.
    """
    feat = np.asarray(feat, dtype=float)
    metric = 'euclidean' if eud else 'cosine'
    C = neighborhood(feat.T, k=k, metric=metric)
    norm_adj = normalized(C.T * C + np.eye(C.shape[0]))
    g = torch.from_numpy(norm_adj).float()
    return g


def normalized(wmat):
    """Symmetric normalisation W = D^-0.5 * W * D^-0.5 (degree = out-degree)."""
    # 计算度矩阵D（出度）
    deg = np.sum(wmat, axis=0)
    # 处理孤立节点：将度为0的位置临时替换为1，避免除以零
    deg_safe = np.where(deg == 0, 1, deg)
    degpow = np.diag(np.power(deg_safe, -0.5))

    # 将孤立节点的归一化系数恢复为0
    degpow[deg == 0, :] = 0
    degpow[:, deg == 0] = 0

    # W = D^-0.5 * W * D^-0.5
    W = np.dot(np.dot(degpow, wmat), degpow)
    return W


# 根据邻接矩阵计算边索引
def adjacency_to_edge_index(adj_matrix):
    rows, cols = torch.nonzero(adj_matrix, as_tuple=True)
    edge_index = torch.stack([rows, cols], dim=0)
    return edge_index
