import torch.utils.data as data
import torch
import h5py
from collections import namedtuple

## COCO-data
paths_COCO = {
    'COCO_train': 'Data/COCO/coco_train.mat',
    'COCO_query': 'Data/COCO/coco_query.mat',
    'COCO_retrieval': 'Data/COCO/coco_retrieval.mat'
}

dataset_lite = namedtuple('dataset_lite', ['img_feature', 'txt_feature', 'label'])

def load_coco(mode):
    if mode == 'train':
        data = h5py.File(paths_COCO['COCO_train'], mode='r')
        img_feature = data['I_tr'][:].T
        txt_feature = data['T_tr'][:].T
        label = data['L_tr'][:].T

    elif mode == 'retrieval':
        data = h5py.File(paths_COCO['COCO_retrieval'], mode='r')
        img_feature = data['I_db'][:].T
        txt_feature = data['T_db'][:].T
        label = data['L_db'][:].T

    else:
        data = h5py.File(paths_COCO['COCO_query'], mode='r')
        img_feature = data['I_te'][:].T
        txt_feature = data['T_te'][:].T
        label = data['L_te'][:].T

    return dataset_lite(img_feature, txt_feature, label)


class my_dataset(data.Dataset):
    def __init__(self, img_feature, txt_feature, label):
        self.img_feature = torch.Tensor(img_feature)
        self.txt_feature = torch.Tensor(txt_feature)
        self.label = torch.Tensor(label)
        self.length = self.img_feature.size(0)

    def __getitem__(self, item):
        return item, self.img_feature[item, :], self.txt_feature[item, :], self.label[item, :]

    def __len__(self):
        return self.length

class my_dataset_str(data.Dataset):
    def __init__(self, img_feature, txt_feature, label, Hash_tea):
        self.img_feature = torch.Tensor(img_feature)
        self.txt_feature = torch.Tensor(txt_feature)
        self.Hash_tea = torch.Tensor(Hash_tea)
        self.label = torch.Tensor(label)
        self.length = self.img_feature.size(0)

    def __getitem__(self, item):
        return item, self.img_feature[item, :], self.txt_feature[item, :], self.label[item, :], self.Hash_tea[item, :]

    def __len__(self):
        return self.length
