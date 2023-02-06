import numpy as np
import os
import pdb
# import tensorflow as tf


from scipy.io import loadmat
from vgg19 import VGG19
# from PIL import Image
from load_data import loading_data
import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

vgg_pretrain = '/data/WangBoWen/PreTrain/vgg/vgg19.npy'


# vgg19 extract image feature
def extract_features(image_data):
    dropoutPro = 1
    classNum = 1000
    skip = []

    image_list = ['train', 'query', 'retrieval']

    # num_data = 20015
    batch_size = 256

    images = tf.placeholder("float", [None, 224, 224, 3])
    model = VGG19(images, dropoutPro, classNum, skip, modelPath=vgg_pretrain)
    feats = model.fc7

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.loadModel(sess)

        feat_all = {'train': [], 'query': [], 'retrieval': []}
        for name in image_list:
            num_data = image_data[name].shape[0]
            index = np.linspace(0, num_data - 1, num_data).astype(int)
            # img_idxs = []
            for i in range(num_data // batch_size + 1):
                # print(i)
                ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
                # print(type(ind))
                images_batch = image_data[name][ind, :, :, :]
                # images_batch = read_image(images_path_batch)
                # print(images_batch.shape)
                # pdb.set_trace()
                feat_I = feats.eval(feed_dict={images: images_batch})

                feat_all[name].append(feat_I)
                # img_idxs.append(ind)
            feat_all[name] = np.concatenate(feat_all[name], axis=0)
    # img_idxs = np.concatenate(img_idxs, axis=0)
    return feat_all


def get_dict(keys, values):
    dictionary = {}
    num_sam = keys.shape[0]
    for i in range(num_sam):
        # dictionary[keys[i][0][0]] = values[i]
        dictionary[keys[i]] = values[i]
    print(len(dictionary))
    return dictionary


if __name__ == '__main__':
    # texts_data, labels_data, image_names = read_mirflickr()
    # img_feats, img_idxs = extract_features(image_names)

    image_data, texts_data, labels_data = loading_data('flickr')

    img_feats = extract_features(image_data)

    # train
    imgs_train = img_feats['train']
    text_train = texts_data['train']
    labels_train = labels_data['train']

    # query
    imgs_query = img_feats['query']
    text_query = texts_data['query']
    labels_query = labels_data['query']

    # retrieval
    imgs_retrieval = img_feats['retrieval']
    text_retrieval = texts_data['retrieval']
    labels_retrieval = labels_data['retrieval']

    train_img_path_list = np.arange(0, imgs_train.shape[0])
    test_img_path_list = np.arange(train_img_path_list[-1] + 1, train_img_path_list[-1] + 1 + imgs_query.shape[0])
    retrieval_img_path_list = np.arange(test_img_path_list[-1] + 1,
                                        test_img_path_list[-1] + 1 + imgs_retrieval.shape[0])

    # index_all = np.random.permutation(20015)
    # ind_Q = index_all[0:2000]
    # ind_T = index_all[2000:20015]
    #
    # # query
    # texts_q = texts_data[img_idxs][ind_Q]
    # imgs_q = img_feats[ind_Q]
    # imgs_path_q = image_names[img_idxs][ind_Q]
    # labels_q = labels_data[img_idxs][ind_Q]
    #
    # # train
    # texts_train = texts_data[img_idxs][ind_T]
    # imgs_train = img_feats[ind_T]
    # imgs_path_train = image_names[img_idxs][ind_T]
    # labels_train = labels_data[img_idxs][ind_T]
    train_tags_dict = get_dict(train_img_path_list, text_train)
    train_labels_dict = get_dict(train_img_path_list, labels_train)
    train_imgs_dict = get_dict(train_img_path_list, imgs_train)

    test_tags_dict = get_dict(test_img_path_list, text_query)  # dict =  {'name':value}
    test_labels_dict = get_dict(test_img_path_list, labels_query)
    test_imgs_dict = get_dict(test_img_path_list, imgs_query)

    retrieval_tags_dict = get_dict(retrieval_img_path_list, text_retrieval)
    retrieval_labels_dict = get_dict(retrieval_img_path_list, labels_retrieval)
    retrieval_imgs_dict = get_dict(retrieval_img_path_list, imgs_retrieval)
    # test_imgs_dict = get_dict(imgs_path_q, imgs_q)
    #
    # train_img_path_list, test_img_path_list = [], []
    # for i in range(len(imgs_path_train)):
    #     train_img_path_list.append(imgs_path_train[i][0][0])
    # for i in range(len(imgs_path_q)):
    #     test_img_path_list.append(imgs_path_q[i][0][0])
    #
    # #print(texts_q.shape, imgs_q.shape, labels_q.shape)
    # print(len(test_tags_dict), len(test_labels_dict), len(train_tags_dict), len(train_labels_dict))
    # # pdb.set_trace()
    #

    np.save('../../data/mir/train_tags_dict.npy', train_tags_dict)
    np.save('../../data/mir/train_imgs_dict.npy', train_imgs_dict)
    np.save('../../data/mir/train_labels_dict.npy', train_labels_dict)
    np.save('../../data/mir/train_img_path_list.npy', train_img_path_list)

    np.save('../../data/mir/test_tags_dict.npy', test_tags_dict)  # text dict={'name': value}
    np.save('../../data/mir/test_imgs_dict.npy', test_imgs_dict)  # image dict = {'name': value}
    np.save('../../data/mir/test_labels_dict.npy', test_labels_dict)  # label dict = {'name': value}
    np.save('../../data/mir/test_img_path_list.npy', test_img_path_list)  # name list=['name1', 'name2', ...]

    np.save('../../data/mir/retrieval_tags_dict.npy', retrieval_tags_dict)  # text dict={'name': value}
    np.save('../../data/mir/retrieval_imgs_dict.npy', retrieval_imgs_dict)  # image dict = {'name': value}
    np.save('../../data/mir/retrieval_labels_dict.npy', retrieval_labels_dict)  # label dict = {'name': value}
    np.save('../../data/mir/retrieval_img_path_list.npy', retrieval_img_path_list)  # name list=['name1', 'name2', ...]
