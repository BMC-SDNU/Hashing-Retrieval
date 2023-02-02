import argparse
import time
import sys
import torch.nn.functional as F
import scipy.io as sio

from model_transformer import GMMH, L2H_Prototype
from utils import *
from data import *

def train(args, dset):
    print('=' * 30)
    print('Training Stage...')
    print('Train size: %d' % (dset.I_tr.shape[0]))
    assert dset.I_tr.shape[0] == dset.T_tr.shape[0]
    assert dset.I_tr.shape[0] == dset.L_tr.shape[0]

    ## Defination[Check in 2022-1-3]
    loss_l2 = torch.nn.MSELoss()
    loss_cl = torch.nn.MultiLabelSoftMarginLoss()

    l2h = L2H_Prototype(args=args)
    l2h.train().cuda()

    gmmh = GMMH(args=args)
    gmmh.train().cuda()

    optimizer_L2H = torch.optim.Adam(l2h.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam([{'params': gmmh.parameters(), 'lr': args.lr}])

    start_time = time.time() * 1000

    ## Preprocess[Check in 2022-1-3]
    _, COO_matrix = get_COO_matrix(dset.L_tr)
    COO_matrix = torch.Tensor(COO_matrix).cuda()
    train_label = torch.Tensor(dset.L_tr).cuda()

    ## Stage 1: learn the hash codes[Check in 2022-1-3]
    for epoch in range(args.epochs_pre):
        prototype, code, pred = l2h(train_label)

        optimizer_L2H.zero_grad()
        B = torch.sign(code)
        prototype_norm = F.normalize(prototype)

        recon_loss = loss_l2(torch.sigmoid(pred), train_label) * args.param_recon_pre
        sign_loss = loss_l2(code, B) * args.param_sign_pre
        bal_loss = torch.sum(code) / code.size(0) * args.param_bal_pre
        static_loss = loss_l2(prototype_norm.mm(prototype_norm.t()), COO_matrix) * args.param_static_pre

        loss = recon_loss + sign_loss + bal_loss + static_loss

        loss.backward()
        optimizer_L2H.step()

    l2h.eval() # to evaluate
    B_tr = np.sign(l2h(train_label)[1].data.cpu().numpy()) # binary

    map_train = calculate_map(B_tr, B_tr, dset.L_tr, dset.L_tr)
    print('Training MAP: %.4f' % (map_train))
    print('=' * 30)

    ## Stage 2: learn the hash functions(Transformer)[Check in 2022-1-3]
    train_loader = data.DataLoader(my_dataset(dset.I_tr, dset.T_tr, dset.L_tr, B_tr=B_tr),
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   pin_memory=True)

    for epoch in range(args.epochs):
        for i, (idx, img_feat, txt_feat, label, B_gnd) in enumerate(train_loader):
            _, aff_norm, aff_label = affinity_tag_multi(label.numpy(), label.numpy())

            img_feat = img_feat.cuda()
            txt_feat = txt_feat.cuda()
            label = label.cuda()
            B_gnd = B_gnd.cuda()

            aff_label = torch.Tensor(aff_label).cuda()

            optimizer.zero_grad()
            H, pred = gmmh(img_feat, txt_feat)
            H_norm = F.normalize(H)

            clf_loss = loss_l2(torch.sigmoid(pred), label)
            sign_loss = loss_l2(H, B_gnd)
            similarity_loss = loss_l2(H_norm.mm(H_norm.t()), aff_label)

            loss = clf_loss * args.param_clf + sign_loss * args.param_sign + similarity_loss * args.param_sim

            loss.backward()
            optimizer.step()

            if (i + 1) == len(train_loader) and (epoch + 1) % 2 == 0:
                print('Epoch [%3d/%3d], Loss: %.4f, Loss-C: %.4f, Loss-B: %.4f, Loss-S: %.4f'
                      % (epoch + 1, args.epochs, loss.item(),
                         clf_loss.item() * args.param_clf,
                         sign_loss.item() * args.param_sign,
                         similarity_loss.item() * args.param_sim))

    end_time = time.time() * 1000
    elapsed = (end_time - start_time) / 1000
    print('Training Time: ', elapsed)
    return gmmh

def eval(model, dset, args):
    model.eval()
    ## Retrieval[Check in 2022-1-3]
    print('=' * 30)
    print('Testing Stage...')
    print('Retrieval size: %d' % (dset.I_db.shape[0]))
    assert dset.I_db.shape[0] == dset.T_db.shape[0]
    assert dset.I_db.shape[0] == dset.L_db.shape[0]

    retrieval_loader = data.DataLoader(my_dataset(dset.I_db, dset.T_db, dset.L_db),
                                       batch_size=args.eval_batch_size,
                                       shuffle=False,
                                       num_workers=0,
                                       pin_memory=True)

    retrievalP = []
    start_time = time.time() * 1000
    for i, (idx, img_feat, txt_feat, _) in enumerate(retrieval_loader):
        img_feat = img_feat.cuda()
        txt_feat = txt_feat.cuda()
        H, _ = model(img_feat, txt_feat)
        retrievalP.append(H.data.cpu().numpy())

    retrievalH = np.concatenate(retrievalP)
    retrievalCode = np.sign(retrievalH)

    end_time = time.time() * 1000
    retrieval_time = end_time - start_time

    ## Query[Check in 2022-1-4]
    print('Query size: %d' % (dset.I_te.shape[0]))
    assert dset.I_te.shape[0] == dset.T_te.shape[0]
    assert dset.I_te.shape[0] == dset.L_te.shape[0]

    val_loader = data.DataLoader(my_dataset(dset.I_te, dset.T_te, dset.L_te),
                                 batch_size=args.eval_batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=True)

    valP = []
    start_time = time.time() * 1000
    for i, (idx, img_feat, txt_feat, _) in enumerate(val_loader):
        img_feat = img_feat.cuda()
        txt_feat = txt_feat.cuda()
        H, _ = model(img_feat, txt_feat)
        valP.append(H.data.cpu().numpy())

    valH = np.concatenate(valP)
    valCode = np.sign(valH)

    end_time = time.time() * 1000
    query_time = end_time - start_time
    print('[Retrieval time] %.4f, [Query time] %.4f' % (retrieval_time / 1000, query_time / 1000))

    # [Check in 2022-1-3]
    if args.save_flag:
        ## Save
        _dict = {
            'retrieval_B': retrievalCode.astype(np.int8),
            'val_B': valCode.astype(np.int8),
            'cateTrainTest': np.sign(dset.L_db @ dset.L_te.T).astype(np.int8),
            'L_db': dset.L_db,
            'L_te': dset.L_te
        }
        sava_path = 'Hashcode/GMMH_' + str(args.nbit) + '_' + args.dataset + '_bits.mat'
        sio.savemat(sava_path, _dict)
    else:
        return retrievalCode, valCode, dset.L_db, dset.L_te.T

    return 0


if __name__ == '__main__':
    # [Check in 2022-1-3]
    parser = argparse.ArgumentParser()

    ## Net basic params
    parser.add_argument('--model', type=str, default='GMMH', help='Use GMMH.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of student epochs to train.')
    parser.add_argument('--epochs_pre', type=int, default=100, help='Epoch to learn the hashcode.')
    parser.add_argument('--nbit', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.5, help='')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--eval_batch_size', type=int, default=512)

    ## Transformer params
    parser.add_argument('--nhead', type=int, default=1, help='"nhead" in Transformer.')
    parser.add_argument('--num_layer', type=int, default=2, help='"num_layer" in Transformer.')
    parser.add_argument('--trans_act', type=str, default='gelu', help='"activation" in Transformer.')

    ## Data params
    parser.add_argument('--dataset', type=str, default='flickr', help='coco/nuswide/flickr')
    parser.add_argument('--classes', type=int, default=24)
    parser.add_argument('--image_dim', type=int, default=4096)
    parser.add_argument('--text_dim', type=int, default=1386)

    ## Net latent dimension params
    # COCO: 128 Flickr: 256
    parser.add_argument('--img_hidden_dim', type=list, default=[2048, 128], help='Construct imageMLP')
    parser.add_argument('--txt_hidden_dim', type=list, default=[1024, 128], help='Construct textMLP')
    parser.add_argument('--L2H_hidden_dim', type=list, default=[1024, 1024], help='Construct L2H')

    ## Loss params
    parser.add_argument('--param_recon_pre', type=float, default=0.001)
    parser.add_argument('--param_sign_pre', type=float, default=100)
    parser.add_argument('--param_bal_pre', type=float, default=0.01)
    parser.add_argument('--param_static_pre', type=float, default=1)

    parser.add_argument('--param_clf', type=float, default=1, help='')
    parser.add_argument('--param_sign', type=float, default=0.01, help='nuswide: 0.0001/')
    parser.add_argument('--param_sim', type=float, default=1)

    ## Flag params
    parser.add_argument('--save_flag', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=2021)
    args = parser.parse_args()

    seed_setting(args.seed)

    dset = load_data(args.dataset)
    print('Train size: %d, Retrieval size: %d, Query size: %d' % (dset.I_tr.shape[0], dset.I_db.shape[0], dset.I_te.shape[0]))
    print('Image dimension: %d, Text dimension: %d, Label dimension: %d' % (dset.I_tr.shape[1], dset.T_tr.shape[1], dset.L_tr.shape[1]))

    args.image_dim = dset.I_tr.shape[1]
    args.text_dim = dset.T_tr.shape[1]
    args.classes = dset.L_tr.shape[1]

    args.img_hidden_dim.insert(0, args.image_dim)
    args.txt_hidden_dim.insert(0, args.text_dim)
    args.L2H_hidden_dim.insert(0, args.classes)
    args.L2H_hidden_dim.append(args.nbit)

    model = train(args, dset)
    eval(model, dset, args)

