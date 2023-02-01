import numpy as np
import torch
import random
import os

def zero2eps(x):
    x[x == 0] = 1
    return x

def normalize(affinity):
    col_sum = zero2eps(np.sum(affinity, axis=1)[:, np.newaxis])
    row_sum = zero2eps(np.sum(affinity, axis=0))
    out_affnty = affinity/col_sum # row data sum = 1
    in_affnty = np.transpose(affinity/row_sum) # col data sum = 1 then transpose
    return in_affnty, out_affnty

def affinity_tag_multi(tag1: np.ndarray, tag2: np.ndarray):

    aff = np.matmul(tag1, tag2.T)
    affinity_matrix = np.float32(aff)
    affinity_matrix = 1 / (1 + np.exp(-affinity_matrix))
    affinity_matrix = 2 * affinity_matrix - 1
    in_aff, out_aff = normalize(affinity_matrix)

    return in_aff, out_aff, affinity_matrix

def seed_setting(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

