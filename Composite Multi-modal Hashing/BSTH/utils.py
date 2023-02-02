import os
import random
import numpy as np
import torch

# Check in 2022-1-3
def seed_setting(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False # False make training process too slow!
    torch.backends.cudnn.deterministic = True

# Refer
def zero2eps(x):
    x[x == 0] = 1
    return x

# Refer
def normalize(affinity):
    col_sum = zero2eps(np.sum(affinity, axis=1)[:, np.newaxis])
    row_sum = zero2eps(np.sum(affinity, axis=0))
    out_affnty = affinity/col_sum # row data sum = 1
    in_affnty = np.transpose(affinity/row_sum) # col data sum = 1 then transpose
    return in_affnty, out_affnty

# Check in 2022-1-3
def affinity_tag_multi(tag1: np.ndarray, tag2: np.ndarray):
    '''
    Use label or plabel to create the graph.
    :param tag1:
    :param tag2:
    :return:
    '''
    aff = np.matmul(tag1, tag2.T)
    affinity_matrix = np.float32(aff)
    # affinity_matrix[affinity_matrix > 1] = 1
    affinity_matrix = 1 / (1 + np.exp(-affinity_matrix))
    affinity_matrix = 2 * affinity_matrix - 1
    in_aff, out_aff = normalize(affinity_matrix)

    return in_aff, out_aff, affinity_matrix

# Refer
def calculate_map(qu_B, re_B, qu_L, re_L):
    """
    :param qu_B: {-1,+1}^{mxq} query bits
    :param re_B: {-1,+1}^{nxq} retrieval bits
    :param qu_L: {0,1}^{mxl} query label
    :param re_L: {0,1}^{nxl} retrieval label
    :return:
    """
    num_query = qu_L.shape[0]
    map = 0
    for iter in range(num_query):
        # gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        count = np.linspace(1, tsum, tsum) # [1,2, tsum]
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        map = map + map_
    map = map / num_query
    return map

# Refer
def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    leng = B2.shape[1] # max inner product value
    distH = 0.5 * (leng - np.dot(B1, B2.transpose()))
    return distH

def get_COO_matrix(label_matrix: np.ndarray, theshold=None)->np.ndarray:
    instance_num, label_dim = label_matrix.shape
    if theshold is None:
        theshold = instance_num // 20
    adj = label_matrix.T @ label_matrix
    adj[range(label_dim), range(label_dim)] = instance_num
    # adj[adj < theshold] = 0
    adj_norm = adj / instance_num
    return adj, adj_norm
