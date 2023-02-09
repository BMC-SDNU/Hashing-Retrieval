import h5py
from PIL import Image
import torch
from torchvision import transforms

# from config import opt
import settings

# RGB
# all_data = h5py.File(opt.DIR, 'r')
# all_data = h5py.File('/data/WangBoWen/HashingRetrieval/feature_data/mir_cnn.mat', 'r')

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

class MY_DATASET(torch.utils.data.Dataset):
    def __init__(self, path, train=True, database=False):
        # self.transform = transform
        # self.target_transform = target_transform
        all_data = h5py.File(path, 'r')
        if train:
            # train
            self.labels = torch.as_tensor(all_data['L_tr'][:].T, dtype=torch.float32)
            self.txt = torch.as_tensor(all_data['T_tr'][:].T, dtype=torch.float32)
            self.images = torch.as_tensor(all_data['I_tr'][:].transpose(3, 2, 0, 1), dtype=torch.float32)
        elif database:
            # retrieval
            self.labels = torch.as_tensor(all_data['L_db'][:].T, dtype=torch.float32)
            self.txt = torch.as_tensor(all_data['T_db'][:].T, dtype=torch.float32)
            self.images = torch.as_tensor(all_data['I_db'][:].transpose(3, 2, 0, 1), dtype=torch.float32)
        else:
            # test
            self.labels = torch.as_tensor(all_data['L_te'][:].T, dtype=torch.float32)
            self.txt = torch.as_tensor(all_data['T_te'][:].T, dtype=torch.float32)
            self.images = torch.as_tensor(all_data['I_te'][:].transpose(3, 2, 0, 1), dtype=torch.float32)

    def __getitem__(self, index):

        # img, target = torch.tensor(self.images[index]), self.train_labels[index]
        # txt = self.txt[index]

        img = torch.as_tensor(self.images[index], dtype=torch.float32)
        target = torch.as_tensor(self.labels[index], dtype=torch.float32)
        txt = torch.as_tensor(self.txt[index], dtype=torch.float32)

        # if self.transform is not None:
        #     img = self.transform(img)
        #
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, txt, target, index

    def __len__(self):
        assert len(self.images) == len(self.labels)
        return len(self.labels)

if __name__ == '__main__':
    data = MY_DATASET()
    print(1111)
    pass