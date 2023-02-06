import h5py
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset

# import settings

# all_data = h5py.File(settings.DIR, 'r')

paths = {
    'flickr': '/data/WangBoWen/HashingRetrieval/feature_data/mir_cnn.mat',
    'nuswide': '../Data/raw_nus.mat',
    'coco': '../Data/raw_coco.mat'
}

def load_dataset(dataname):
    all_data = h5py.File(paths[dataname], 'r')

    return all_data

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),  # HWC+[0,255] -> CHW+[0,1]
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# txt_feat_len = all_data['T_tr'].shape[0]


class MY_DATASET(Dataset):

    def __init__(self, transform=None, target_transform=None, dataname=None, train=True, database=False):
        self.all_data = load_dataset(dataname)
        self.transform = transform
        self.target_transform = target_transform
        if train:
            self.labels = self.all_data['L_tr'][:]
            self.txt = self.all_data['T_tr'][:]
            self.images = self.all_data['I_tr'][:]
        elif database:
            self.labels = self.all_data['L_db'][:]
            self.txt = self.all_data['T_db'][:]
            self.images = self.all_data['I_db'][:]
        else:
            self.labels = self.all_data['L_te'][:]
            self.txt = self.all_data['T_te'][:]
            self.images = self.all_data['I_te'][:]

    def __getitem__(self, index):

        img_feature, target = self.images[index], self.labels[index]
        # img = img[:, :, ::-1].copy()  # BGR -> RGB
        # print('*' * 1000)
        # print(img.shape)

        # img = Image.fromarray(np.transpose(img, (1, 0, 2))) # HWC
        # img = Image.fromarray(img)  # HWC

        txt = self.txt[index]

        # if self.transform is not None:
        #     img = self.transform(img_feature)
        #
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img_feature, txt, target, index

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    data = MY_DATASET(dataname='flickr')
    print(data[0])
    pass