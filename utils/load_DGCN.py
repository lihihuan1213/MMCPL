import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp


def load_resource_index(index_file):
    df = pd.read_csv(index_file, header=None)
    id_to_index = {row[1]: row[0] for _, row in df.iterrows()}
    index_to_id = {row[0]: row[1] for _, row in df.iterrows()}
    num_nodes = len(id_to_index)
    return id_to_index, index_to_id, num_nodes


def generate_adj_matrices(edge_file, index_file):
    id_to_index, index_to_id, num_nodes = load_resource_index(index_file)
    adj_out = np.zeros((num_nodes, num_nodes))
    adj_in = np.zeros((num_nodes, num_nodes))

    edges_df = pd.read_csv(edge_file, header=None)
    source_ids = edges_df[0]
    target_ids = edges_df[1]

    for source_id, target_id in zip(source_ids, target_ids):
        if source_id in id_to_index and target_id in id_to_index:
            source_idx = id_to_index[source_id]
            target_idx = id_to_index[target_id]
            adj_out[source_idx][target_idx] += 1.0
            adj_in[target_idx][source_idx] += 1.0

    adj_out = normalize(adj_out + sp.eye(num_nodes))
    adj_in = normalize(adj_in + sp.eye(num_nodes))

    adj_out = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj_out))
    adj_in = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj_in))
    return adj_out, adj_in


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    # return torch.sparse.FloatTensor(indices, values, shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

