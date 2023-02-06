import numpy as np
import os
import pdb
import tensorflow as tf

from scipy.io import loadmat
from vgg19 import VGG19
from PIL import Image


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def read_mirflickr():
    texts_path = "/.../mirflickr25k-yall.mat"
    labels_path = "/.../mirflickr25k-lall.mat"
    images_path = "/.../mirflickr25k-fall.mat"

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


# vgg19 extract image feature
def extract_features(images_path):
    dropoutPro = 1
    classNum = 1000
    skip = []

    num_data = 20015
    batch_size = 256

    images = tf.placeholder("float", [None, 224, 224, 3])
    model = VGG19(images, dropoutPro, classNum, skip)
    feats = model.fc7

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.loadModel(sess)

        feat_all = []
        index = np.linspace(0, num_data - 1, num_data).astype(int)
        img_idxs = []
        for i in range(num_data // batch_size + 1):
            print(i)
            ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
            images_path_batch = images_path[ind]
            images_batch = read_image(images_path_batch)
            # print(images_batch.shape)
            # pdb.set_trace()
            feat_I = feats.eval(feed_dict={images: images_batch})

            feat_all.append(feat_I)
            img_idxs.append(ind)
    feat_all = np.concatenate(feat_all, axis=0)
    img_idxs = np.concatenate(img_idxs, axis=0)
    return feat_all, img_idxs


def get_dict(keys, values):
    dictionary = {}
    num_sam = keys.shape[0]
    for i in range(num_sam):
        dictionary[keys[i][0][0]] = values[i]
    print(len(dictionary))
    return dictionary


texts_data, labels_data, image_names = read_mirflickr()
img_feats, img_idxs = extract_features(image_names)

index_all = np.random.permutation(20015)
ind_Q = index_all[0:2000]
ind_T = index_all[2000:20015]

# query
texts_q = texts_data[img_idxs][ind_Q]
imgs_q = img_feats[ind_Q]
imgs_path_q = image_names[img_idxs][ind_Q]
labels_q = labels_data[img_idxs][ind_Q]

# train
texts_train = texts_data[img_idxs][ind_T]
imgs_train = img_feats[ind_T]
imgs_path_train = image_names[img_idxs][ind_T]
labels_train = labels_data[img_idxs][ind_T]

test_tags_dict = get_dict(imgs_path_q, texts_q)  # dict =  {'name':value}
test_labels_dict = get_dict(imgs_path_q, labels_q)
test_imgs_dict = get_dict(imgs_path_q, imgs_q)
train_tags_dict = get_dict(imgs_path_train, texts_train)
train_labels_dict = get_dict(imgs_path_train, labels_train)
train_imgs_dict = get_dict(imgs_path_train, imgs_train)
# test_imgs_dict = get_dict(imgs_path_q, imgs_q)

train_img_path_list, test_img_path_list = [], []
for i in range(len(imgs_path_train)):
    train_img_path_list.append(imgs_path_train[i][0][0])
for i in range(len(imgs_path_q)):
    test_img_path_list.append(imgs_path_q[i][0][0])

#print(texts_q.shape, imgs_q.shape, labels_q.shape)
print(len(test_tags_dict), len(test_labels_dict), len(train_tags_dict), len(train_labels_dict))
pdb.set_trace()

np.save('/.../data/test_tags_dict.npy', test_tags_dict)  # text dict={'name': value}
np.save('/.../data/test_imgs_dict.npy', test_imgs_dict)  # image dict = {'name': value}
np.save('/.../data/test_labels_dict.npy', test_labels_dict)  # label dict = {'name': value}
np.save('/.../data/test_img_path_list.npy', test_img_path_list)  # name list=['name1', 'name2', ...]

np.save('/.../data/train_tags_dict.npy', train_tags_dict)
np.save('/.../data/train_imgs_dict.npy', train_imgs_dict)
np.save('/.../data/train_labels_dict.npy', train_labels_dict)
np.save('/.../data/train_img_path_list.npy', train_img_path_list)
