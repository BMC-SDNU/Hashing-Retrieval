from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
# from sklearn import preprocessing
import pdb
import torch

import h5py
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset

# import settings

# all_data = h5py.File(settings.DIR, 'r')

paths = {
    'flickr': '../Data/mir_cnn.mat',
    'nuswide': '../Data/raw_nus.mat',
    'coco': '../Data/raw_coco.mat'
}


class CustomDataSet(Dataset):
    def __init__(self, images, texts, labels):
        self.images = torch.tensor(images, dtype=torch.float32)
        self.texts = torch.tensor(texts, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]

        label = self.labels[index]
        return img, text, label, index

    def __len__(self):
        count = len(self.images)
        # print(len(self.images))
        # print(len(self.labels))
        assert len(self.images) == len(self.labels)
        return count

def load_dataset(dataname):
    dataset = h5py.File(paths[dataname], 'r')

    train_I = dataset['I_tr'][:].T
    train_T = dataset['T_tr'][:].T
    train_L = dataset['L_tr'][:].T

    query_I = dataset['I_te'][:].T
    query_T = dataset['T_te'][:].T
    query_L = dataset['L_te'][:].T

    retrieval_I = dataset['I_db'][:].T
    retrieval_T = dataset['T_db'][:].T
    retrieval_L = dataset['L_db'][:].T


    rand_sample = torch.randperm(n=len(train_L))
    validation_I = train_I[rand_sample[:2000]]
    validation_T = train_T[rand_sample[:2000]]
    validation_L = train_L[rand_sample[:2000]]


    if dataname == 'nus' or dataname == 'coco':
        retrieval_vI = retrieval_I[:10000]
        retrieval_vT = retrieval_T[:10000]
        retrieval_vL = retrieval_L[:10000]
    else:
        retrieval_vI = retrieval_I
        retrieval_vT = retrieval_T
        retrieval_vL = retrieval_L


    imgs = {'train': train_I, 'query': query_I, 'database': retrieval_I, 'databasev':retrieval_vI, 'validation':validation_I}
    texts = {'train': train_T, 'query': query_T, 'database': retrieval_T, 'databasev':retrieval_vT, 'validation':validation_T}
    labels = {'train': train_L, 'query': query_L, 'database': retrieval_L, 'databasev':retrieval_vL, 'validation':validation_L}


    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
               for x in ['query', 'train', 'database', 'databasev', 'validation']}
    return dataset, (train_I, train_T, train_L)




