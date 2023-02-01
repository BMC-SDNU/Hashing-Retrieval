import argparse
import time
import sys
sys.path.append('..')

import scipy.io as sio
from utils.datasets import *
from utils.utils import *
from model import HGCN, Discriminator, StudentImgNet, StudentTxtNet, HashFusion
seed_setting()

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='HGCN', help='Use HGCN.')
parser.add_argument('--tea_epochs', type=int, default=20, help='Number of teacher epochs to train.')
parser.add_argument('--stu_epochs', type=int, default=20, help='Number of student epochs to train.')
parser.add_argument('--nbits', type=int, default=64)
parser.add_argument('--GD_lr', type=float, default=0.0001)
parser.add_argument('--stu_lr', type=float, default=0.0001, help='Initial student network learning rate.')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--dataset', type=str, default='Flickr', help='COCO/NUSWIDE/Flickr')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--n_class', type=int, default=24)
parser.add_argument('--img_dim', type=int, default=4096)
parser.add_argument('--txt_dim', type=int, default=1386)

parser.add_argument('--s_param', type=float, default=100, help='[Teacher]The factor of kS-cos(B, B) from DJSRH loss.')
parser.add_argument('--g_param', type=float, default=1, help='[Teacher]The factor of generator loss.')
parser.add_argument('--b_param', type=float, default=0.001, help='[Teacher]The factor of SIGN loss.')
parser.add_argument('--b_param_stu', type=float, default=1, help='[Student]The factor of SIGN loss.')
parser.add_argument('--s_param_stu', type=float, default=1, help='[Student]The factor of supervise infomation from teacher network.')
parser.add_argument('--emb_lr_factor', type=float, default=10)
parser.add_argument('--fusion_lr_factor', type=float, default=0.01)

args = parser.parse_args()

loss_adv = torch.nn.BCEWithLogitsLoss()
loss_l2 = torch.nn.MSELoss()

def train_teacher(args):

    # 1. load data
    dset = load_coco(mode='train')

    train_loader = data.DataLoader(my_dataset(dset.img_feature, dset.txt_feature, dset.label),
                                   batch_size=args.batch_size,
                                   shuffle=True)

    # 2. define the model
    hgcn, discriminator = HGCN(args=args), Discriminator(args)

    hgcn.cuda(), discriminator.cuda()
    hgcn.train(), discriminator.train()

    optimizer_G = torch.optim.Adam(hgcn.parameters(), lr=args.GD_lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.GD_lr)

    start_time = time.time() * 1000
    for epoch in range(args.tea_epochs):
        for i, (idx, img_feat, txt_feat, label) in enumerate(train_loader):

            _, aff_norm, aff_label = affinity_tag_multi(label.numpy(), label.numpy())
            aff_cross_1 = np.hstack((np.eye(aff_label.shape[0]), aff_label))
            aff_cross_2 = np.hstack((aff_label, np.eye(aff_label.shape[0])))
            aff_cross = np.vstack((aff_cross_1, aff_cross_2))
            # normalize the CM-affinity
            _, aff_cross_norm = normalize(aff_cross)

            now_size = aff_label.shape[0]

            aff_label = Variable(torch.Tensor(aff_label).cuda())
            aff_norm = Variable(torch.Tensor(aff_norm)).cuda()
            aff_cross_norm = Variable(torch.Tensor(aff_cross_norm)).cuda()
            img_feat = Variable(img_feat).cuda()
            txt_feat = Variable(txt_feat).cuda()
            label = Variable(label).cuda()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            hash = hgcn(img_feat, txt_feat, aff_norm, aff_cross_norm, now_size)

            optimizer_D.zero_grad()

            real_contrib = Variable(torch.randn(size=(hash.size()[0], hash.size()[1]))).cuda() # guass

            d_fake = discriminator(hash)
            d_real = discriminator(real_contrib)

            dc_loss_real = loss_l2(d_real, Variable(torch.ones(d_real.shape[0], 1)).cuda())
            dc_loss_fake = loss_l2(d_fake, Variable(torch.zeros(d_fake.shape[0], 1)).cuda())
            loss_D = dc_loss_fake + dc_loss_real

            loss_D.backward()
            optimizer_D.step()
            if i + 1 == len(train_loader) and (epoch+1) % 2 == 0:
                print('Epoch [%3d/%3d], Loss: %.4f, Loss-real: %.4f, Loss-fake: %.4f'
                    % (epoch + 1, args.tea_epochs, loss_D.item(), dc_loss_real.item(), dc_loss_fake.item()))

            # -----------------
            #  Train Generator
            # -----------------
            for j in range(5):
                optimizer_G.zero_grad()

                hash = hgcn(img_feat, txt_feat, aff_norm, aff_cross_norm, now_size)

                hash_norm = F.normalize(hash)
                B = torch.sign(hash)

                d_fake = discriminator(hash)
                generator_loss = loss_l2(d_fake, Variable(torch.ones(d_fake.shape[0], 1)).cuda())
                sign_loss = loss_l2(hash, B)
                construct_loss = loss_l2(hash_norm.mm(hash_norm.t()), aff_label)
                loss_G = generator_loss * args.g_param + sign_loss * args.b_param + construct_loss * args.s_param

                loss_G.backward()
                optimizer_G.step()
                if i + 1 == len(train_loader) and (epoch + 1) % 2 == 0 and j == 4:
                    print('Epoch [%3d/%3d], Loss: %.4f, Loss-S: %.4f, Loss-B: %.4f, Loss-G: %.4f'
                          % (epoch + 1, args.tea_epochs, loss_G.item(), construct_loss.item() * args.s_param, sign_loss.item() * args.b_param, generator_loss.item() * args.g_param))

    end_time = time.time() * 1000
    print('Teacher Training Time: ', (end_time - start_time) / 1000)
    return dset, hgcn

def train_student(dset, hgcn, args):

    imgStudent, txtStudent, fusionModule = StudentImgNet(args), StudentTxtNet(args), HashFusion(args)
    imgStudent.cuda(), txtStudent.cuda(), fusionModule.cuda()

    # Get teacher net infomation
    train_size = dset.img_feature.shape[0]
    Hash_Tea = torch.zeros(size=(train_size, args.nbits))

    hgcn.eval()
    train_loader = data.DataLoader(my_dataset(dset.img_feature, dset.txt_feature, dset.label),
                                   batch_size=args.batch_size,
                                   shuffle=False) 

    start_time_pre = time.time() * 1000
    for i, (idx, img_feat, txt_feat, label) in enumerate(train_loader):

        _, aff_norm, aff_label = affinity_tag_multi(label.numpy(), label.numpy())
        aff_cross_1 = np.hstack((np.eye(aff_label.shape[0]), aff_label))
        aff_cross_2 = np.hstack((aff_label, np.eye(aff_label.shape[0])))
        aff_cross = np.vstack((aff_cross_1, aff_cross_2))
        _, aff_cross_norm = normalize(aff_cross)

        now_size = aff_label.shape[0]

        aff_label = Variable(torch.Tensor(aff_label).cuda())
        aff_norm = Variable(torch.Tensor(aff_norm)).cuda()
        aff_cross_norm = Variable(torch.Tensor(aff_cross_norm)).cuda()
        img_feat = Variable(img_feat).cuda()
        txt_feat = Variable(txt_feat).cuda()

        hash = hgcn(img_feat, txt_feat, aff_norm, aff_cross_norm, now_size)

        Hash_Tea[i * args.batch_size : (i + 1) * args.batch_size, :] = hash
    end_time_pre = time.time() * 1000
    print('Preprocess Student Training Time: ', (end_time_pre - start_time_pre) / 1000)


    embedding_params = list(map(id, txtStudent.Embedding.parameters()))
    base_params = filter(lambda p: id(p) not in embedding_params, txtStudent.parameters())
    optimizer_stu = torch.optim.Adam([{'params': base_params},
                                      {'params': txtStudent.Embedding.parameters(), 'lr': args.stu_lr * args.emb_lr_factor},
                                      {'params': filter(lambda p: p.requires_grad, imgStudent.parameters())},
                                      {'params': fusionModule.parameters(), 'lr': args.stu_lr * args.fusion_lr_factor}],
                                     lr=args.stu_lr)

    ## Distill the student net
    train_loader = data.DataLoader(my_dataset_str(dset.img_feature, dset.txt_feature, dset.label, Hash_Tea),
                                   batch_size=args.batch_size,
                                   shuffle=False)

    imgStudent.train(), txtStudent.train(), fusionModule.train()
    def freeze_bn(L):
        if isinstance(L, nn.BatchNorm1d):
            L.eval()
    start_time = time.time() * 1000
    for epoch in range(args.stu_epochs):
        if epoch >= 5:
            imgStudent.apply(freeze_bn)
            txtStudent.apply(freeze_bn)
        for i, (idx, img_feat, txt_feat, label, hash_tea) in enumerate(train_loader):
            img_feat = Variable(img_feat).cuda()
            txt_feat = Variable(txt_feat).cuda()
            hash_tea = Variable(hash_tea).cuda()

            img_hash = imgStudent(img_feat)
            txt_hash = txtStudent(txt_feat)
            fusion_hash = fusionModule(img_hash, txt_hash)

            optimizer_stu.zero_grad()

            img_stu_norm = F.normalize(img_hash)
            txt_stu_norm = F.normalize(txt_hash)
            fusion_norm = F.normalize(fusion_hash)

            hash_tea_norm = F.normalize(hash_tea)
            img_tea_norm = F.normalize(hash_tea[:, :int(args.nbits / 2)])
            txt_tea_norm = F.normalize(hash_tea[:, int(args.nbits / 2): ])

            B = torch.sign(hash_tea)
            b_loss = loss_l2(img_hash, B) + loss_l2(txt_hash, B) + loss_l2(fusion_hash, B)

            s_loss = loss_l2(img_stu_norm.mm(img_stu_norm.t()), img_tea_norm.mm(img_tea_norm.t())) + \
                     loss_l2(txt_stu_norm.mm(txt_stu_norm.t()), txt_tea_norm.mm(txt_tea_norm.t())) + \
                     loss_l2(fusion_norm.mm(fusion_norm.t()), hash_tea_norm.mm(hash_tea_norm.t())) + \
                     loss_l2(img_stu_norm.mm(txt_stu_norm.t()), hash_tea_norm.mm(hash_tea_norm.t()))
            loss = b_loss * args.b_param_stu + s_loss * args.s_param_stu

            loss.backward()
            optimizer_stu.step()

            if i + 1 == len(train_loader) and (epoch + 1) % 2 == 0:
                print('Epoch [%3d/%3d], Loss: %.4f,  Loss-B: %.4f, Loss-S: %.4f' %
                      (epoch + 1, args.stu_epochs, loss.item(), b_loss.item(), s_loss.item()))

    end_time = time.time() * 1000
    print('Student Training Time: ', (end_time - start_time) / 1000)
    return imgStudent, txtStudent, fusionModule

def test(args, imgStudent, txtStudent, fusionModule, save_flag=True):
    imgStudent.eval(), txtStudent.eval(), fusionModule.eval()

    ## Retrieval
    dset = load_coco(mode='retrieval')
    retrieval_loader = data.DataLoader(my_dataset(dset.img_feature, dset.txt_feature, dset.label),
                                       batch_size=4096,
                                       shuffle=False,
                                       num_workers=0)

    retrievalP_img = []
    retrievalP_txt = []
    retrieval_label = dset.label
    start_time = time.time() * 1000
    for i, (idx, img_feat, txt_feat, label) in enumerate(retrieval_loader):
        img_feat = Variable(img_feat).cuda()
        txt_feat = Variable(txt_feat).cuda()

        img_hash = imgStudent(img_feat)
        txt_hash = txtStudent(txt_feat)
        fusion_hash = fusionModule(img_hash, txt_hash)

        retrievalP_img.append(fusion_hash.data.cpu().numpy())
        retrievalP_txt.append(fusion_hash.data.cpu().numpy())

    retrievalH_img = np.concatenate(retrievalP_img)
    retrievalH_txt = np.concatenate(retrievalP_txt)
    retrievalCode_img = np.sign(retrievalH_img)
    retrievalCode_txt = np.sign(retrievalH_txt)

    end_time = time.time() * 1000
    retrieval_time = end_time - start_time


    dset = load_coco(mode='val')
    val_loader = data.DataLoader(my_dataset(dset.img_feature, dset.txt_feature, dset.label),
                                 batch_size=4096,
                                 shuffle=False,
                                 num_workers=0)
    valP_img = []
    valP_txt = []
    val_label = dset.label
    start_time = time.time() * 1000
    for i, (idx, img_feat, txt_feat, label) in enumerate(val_loader):
        img_feat = Variable(img_feat).cuda()
        txt_feat = Variable(txt_feat).cuda()

        img_hash = imgStudent(img_feat)
        txt_hash = txtStudent(txt_feat)

        valP_img.append(img_hash.data.cpu().numpy())
        valP_txt.append(txt_hash.data.cpu().numpy())

    valH_img = np.concatenate(valP_img)
    valH_txt = np.concatenate(valP_txt)
    valCode_img = np.sign(valH_img)
    valCode_txt = np.sign(valH_txt)

    end_time = time.time() * 1000
    query_time = end_time - start_time

    print('[Retrieval time] %.4f, [Query time] %.4f' % (retrieval_time / 1000, query_time / 1000))

    if save_flag:
        ## Save
        _dict = {
            'retrieval_img_B': retrievalCode_img,
            'retrieval_txt_B': retrievalCode_txt,
            'val_img_B': valCode_img,
            'val_txt_B': valCode_txt,
            'cateTrainTest': np.sign(retrieval_label @ val_label.T)
        }

        sava_path = 'Hashcode/GCN_' + str(args.nbits) + '_' + args.dataset + '_bits.mat'
        sio.savemat(sava_path, _dict)
    else:
        return retrievalCode_img, retrievalCode_txt, valCode_img, valCode_txt, retrieval_label, val_label

    return 0

if __name__ == '__main__':
    if args.model == 'HGCN':
        dset, model = train_teacher(args)
        imgStudent, txtStudent, fusionModule = train_student(dset, model, args)
        test(args, imgStudent, txtStudent, fusionModule)
    else:
        print('No model can be used!')
