import os
import tensorflow as tf
import pdb
import numpy as np
import time
import inspect
from read_nuswide import loading_data, read_images

from resnet50 import ResNet
from keras.layers import Flatten, Input
from keras.models import Model

VGG_MEAN = [103.939, 116.779, 123.68]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
class Vgg19:
    def __init__(self, vgg19_npy_path=None):
        if vgg19_npy_path is None:
            path = inspect.getfile(Vgg19)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg19.npy")
            vgg19_npy_path = path
            print(vgg19_npy_path)

        self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def build(self, rgb):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        #print("build model started")
        #rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

#         self.fc8 = self.fc_layer(self.relu7, "fc8")

#         self.prob = tf.nn.softmax(self.fc8, name="prob")

        #self.data_dict = None
        #print(("build model finished: %ds" % (time.time() - start_time)))
        return self.relu7

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")


def get_dict(keys, values):
    dictionary = {}
    num_sam = keys.shape[0]
    for i in range(num_sam):
        dictionary[keys[i][0][0]] = values[i]
    print(len(dictionary))
    return dictionary


def get_feat(sess, train_img_path, num_data):
    model_para_path = '/cache/vgg19.npy'
    VGG_model = Vgg19(vgg19_npy_path=model_para_path)

    ph = {}
    ph['image_input'] = tf.placeholder(tf.float32, [None, 224, 224, 3], name='image_input')

    img_feats = VGG_model.build(ph['image_input'])

    batch_size = 256
    imgs_feat, imgs_path = [], []
    index = np.random.permutation(num_data)
    #with tf.Session() as sess:
    for iter in range(num_data // batch_size+1):
        print('iter', iter)
        ind = index[iter * batch_size: min((iter + 1) * batch_size, num_data)]
        imgPath_batch = train_img_path[ind]
        imgs_batch = read_images(imgPath_batch).astype(np.float32)
        #print(imgs_batch.shape)

        img_feat = sess.run(img_feats, feed_dict={ph['image_input']: imgs_batch})
        #img_feat = VGG_model.build(imgs_batch)
        #print(img_feat)
        #feature_dict = dict(zip(imgPath_batch, img_feat))
        imgs_feat.append(img_feat)
        imgs_path.append(imgPath_batch)
    return np.array(imgs_feat), np.array(imgs_path)

# def get_feat(sess, train_img_path, num_data):
#     DATABASE_SIZE = 18015
#     batch_size = 256
#     imgs_feat, imgs_path = [], []
#     index = np.random.permutation(num_data)
#     #with tf.Session() as sess:
#     for iter in range(num_data // batch_size+1):
#         print('iter', iter)
#         ind = index[iter * batch_size: min((iter + 1) * batch_size, num_data)]
#         imgPath_batch = train_img_path[ind]
#         imgs_batch = read_images(imgPath_batch).astype(np.float32)

#         img_feat = sess.run(img_feats, feed_dict={ph['image_input']: imgs_batch})
# #         print(img_feat.shape)
# #         pdb.set_trace()

#         imgs_feat.append(img_feat)
#         imgs_path.append(imgPath_batch)
#     return np.array(imgs_feat), np.array(imgs_path)


images_path, tags, labels = loading_data()
print(images_path.shape, tags.shape, labels.shape)

DATABASE_SIZE = 184712
QUERY_SIZE = 1865
index_all = np.random.permutation(QUERY_SIZE+DATABASE_SIZE)
ind_Q = index_all[0:QUERY_SIZE]
ind_R = index_all[QUERY_SIZE:DATABASE_SIZE + QUERY_SIZE]

test_img_path = images_path[ind_Q]
test_tags = tags[ind_Q]
test_labels = labels[ind_Q]
train_img_path = images_path[ind_R]
train_tags = tags[ind_R]
train_labels = labels[ind_R]

num_train_img = len(train_img_path)
num_test_img = len(test_img_path)
train_img_path_list, test_img_path_list = [], []
for i in range(num_train_img):
    train_img_path_list.append(train_img_path[i][0][0])
for i in range(num_test_img):
    test_img_path_list.append(test_img_path[i][0][0])

#pdb.set_trace()
workdir = '/cache/'
np.save(workdir + 'nus_train_img_path_list.npy', train_img_path_list)
np.save(workdir + 'nus_test_img_path_list.npy', test_img_path_list)

test_tags_dict = get_dict(test_img_path, test_tags)
test_labels_dict = get_dict(test_img_path, test_labels)
train_tags_dict = get_dict(train_img_path, train_tags)
train_labels_dict = get_dict(train_img_path, train_labels)

with tf.Session() as sess:
    train_imgs_feat, train_imgs_path = get_feat(sess, train_img_path, DATABASE_SIZE)
    test_imgs_feat, test_imgs_path = get_feat(sess, test_img_path, QUERY_SIZE)
# with tf.Session() as sess:
#     pretr_weights_path = '/cache/weights_resnet.npy'
#     resnet = ResNet(resnet_npy_path = pretr_weights_path)

#     ph = {}
#     ph['image_input'] = tf.placeholder(tf.float32, [None, 224, 224, 3], name='image_input')

#     #img_feats = VGG_model.build(ph['image_input'])
#     resnet.build(ph['image_input'])
#     img_feats = resnet.fc1

#     sess.run(tf.global_variables_initializer())
#     resnet.load_weights(sess)

#     train_imgs_feat, train_imgs_path = get_feat(sess, train_img_path, DATABASE_SIZE)  
#     test_imgs_feat, test_imgs_path = get_feat(sess, test_img_path, QUERY_SIZE)
#print(imgs_feat[0:5])
train_imgs_feat, train_imgs_path = np.concatenate(train_imgs_feat), np.concatenate(train_imgs_path)
test_imgs_feat, test_imgs_path = np.concatenate(test_imgs_feat), np.concatenate(test_imgs_path)
train_imgs_dict = get_dict(train_imgs_path, train_imgs_feat)
test_imgs_dict = get_dict(test_imgs_path, test_imgs_feat)
print(train_imgs_feat.shape, train_imgs_path.shape)
print(test_imgs_feat.shape, test_imgs_path.shape)

np.save(workdir + 'nus_train_img_dict.npy', train_imgs_dict)
np.save(workdir + 'nus_train_txts_dict.npy', train_tags_dict)
np.save(workdir + 'nus_train_labels_dict.npy', train_labels_dict)

np.save(workdir + 'nus_test_img_dict.npy', test_imgs_dict)
np.save(workdir + 'nus_test_txts_dict.npy', test_tags_dict)
np.save(workdir + 'nus_test_labels_dict.npy', test_labels_dict)
print('Done')
