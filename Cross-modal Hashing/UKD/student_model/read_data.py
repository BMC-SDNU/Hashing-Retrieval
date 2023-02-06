import numpy as np
import os
import pdb
import tensorflow as tf

from scipy.io import loadmat
#from vgg19 import VGG19
from PIL import Image


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def read_mirflickr():
    texts_path = "/home/huhengtong/UKD/teacher_UGACH/KNN/mirflickr25k-yall.mat"
    labels_path = "/home/huhengtong/UKD/teacher_UGACH/KNN/mirflickr25k-lall.mat"
    images_path = "/home/huhengtong/UKD/teacher_UGACH/KNN/mirflickr25k-fall.mat"

    texts_data = loadmat(texts_path)['YAll']
    labels_data = loadmat(labels_path)['LAll']
    image_names = loadmat(images_path)['FAll']

    #print(data['FAll'])
    return texts_data, labels_data, image_names


def read_image(paths):
    prefix_path = '/home/huhengtong/data/mirflickr'
    images = []
    for elem in paths:
        # print(elem[0][0])
        # pdb.set_trace()
        img_path = os.path.join(prefix_path, elem[0][0])
        img = Image.open(img_path)
        img = img.resize((256, 256))
        img_c = img.crop((16, 16, 240, 240))
        # print(np.array(img_c).shape)
        # pdb.set_trace()
        images.append(np.array(img_c))
    return np.stack(images, axis=0)



# def extract_features(images_path):
#     dropoutPro = 1
#     classNum = 1000
#     skip = []
#
#     num_data = 20015
#     batch_size = 256
#
#     images = tf.placeholder("float", [None, 224, 224, 3])
#     model = VGG19(images, dropoutPro, classNum, skip)
#     feats = model.fc7
#
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         model.loadModel(sess)
#
#         feat_all = []
#         index = np.linspace(0, num_data - 1, num_data).astype(int)
#         img_idxs = []
#         for i in range(num_data // batch_size + 1):
#             print(i)
#             ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
#             images_path_batch = images_path[ind]
#             images_batch = read_image(images_path_batch)
#             # print(images_batch.shape)
#             # pdb.set_trace()
#             feat_I = feats.eval(feed_dict={images: images_batch})
#
#             feat_all.append(feat_I)
#             img_idxs.append(ind)
#     feat_all = np.concatenate(feat_all, axis=0)
#     img_idxs = np.concatenate(img_idxs, axis=0)
#     return feat_all, img_idxs


# texts_data, labels_data, image_names = read_mirflickr()
# img_feats, img_idxs = extract_features(image_names)
#
# index_all = np.random.permutation(20015)
# ind_Q = index_all[0:2000]
# ind_T = index_all[2000:20015]
#
# texts_q = texts_data[img_idxs][ind_Q]
# imgs_q = img_feats[ind_Q]
# labels_q = labels_data[img_idxs][ind_Q]
#
# texts_train = texts_data[img_idxs][ind_T]
# imgs_train = img_feats[ind_T]
# labels_train = labels_data[img_idxs][ind_T]
# print(texts_q.shape, imgs_q.shape, labels_q.shape)
# print(texts_train.shape, imgs_train.shape, labels_train.shape)
# pdb.set_trace()
#
# np.save('/home/huhengtong/UKD/texts_q.npy', texts_q)
# np.save('/home/huhengtong/UKD/imgs_q.npy', imgs_q)
# np.save('/home/huhengtong/UKD/labels_q.npy', labels_q)
#
# np.save('/home/huhengtong/UKD/texts_train.npy', texts_train)
# np.save('/home/huhengtong/UKD/imgs_train.npy', imgs_train)
# np.save('/home/huhengtong/UKD/labels_train.npy', labels_train)