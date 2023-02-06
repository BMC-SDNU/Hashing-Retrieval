import pickle, random, pdb
import scipy.io as sio
import os
# import tensorflow as tf
import numpy as np
import utils as ut
from map import *
from dis_model_nn import DIS
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

IMAGE_DIM = 4096
TEXT_DIM = 1386
OUTPUT_DIM = 128
HIDDEN_DIM = 1024
CLASS_DIM = 24
WEIGHT_DECAY = 0.01
D_LEARNING_RATE = 0.001
G_LEARNING_RATE = 0.001
TEMPERATURE = 0.2
BETA = 5.0
GAMMA = 0.1

WORKDIR = '/....../'
#DIS_MODEL_BEST_FILE = WORKDIR + 'teacher_best_' + str(OUTPUT_DIM) + '.model'
DIS_MODEL_BEST_I2I_FILE = '......'

train_img = np.load('/....../imgs_train.npy')
train_txt = np.load('/....../texts_train.npy')
train_label = np.load('/....../labels_train.npy')


def extract_feature(sess, model, data, flag):
	num_data = len(data)
	batch_size = 256
	index = np.linspace(0, num_data - 1, num_data).astype(np.int32)

	feat_data = []
	for i in range(num_data // batch_size + 1):
		ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]

		data_batch = data[ind]
		if flag == 'image':
			output_feat = sess.run(model.image_sig, feed_dict={model.image_data: data_batch})
		elif flag == 'text':
			output_feat = sess.run(model.text_sig, feed_dict={model.text_data: data_batch})
		feat_data.append(output_feat)
	feat_data = np.concatenate(feat_data)
		
	return feat_data


def get_AP(k_nearest, label, query_index, k):
    score = 0.0
    for i in range(k):
        if np.dot(label[query_index], label[int(k_nearest[i])]) > 0:
            score += 1.0
    return score / k


def get_knn(img_feat, txt_feat):
	train_size = 18015
	K = 10000
	KNN_img = np.zeros((train_size, K))
	KNN_txt = np.zeros((train_size, K))
	accuracy_sum_img = 0
	accuracy_sum_txt = 0

	distance_img = pdist(img_feat, 'euclidean')
	distance_txt = pdist(txt_feat, 'euclidean')

	distance_img = squareform(distance_img)
	distance_txt = squareform(distance_txt)

	for i in range(train_size):
		k_nearest_img = np.argsort(distance_img[i])[0:K]
		k_nearest_txt = np.argsort(distance_txt[i])[0:K]

		accuracy_sum_img += get_AP(k_nearest_img, train_label, i, K)
		accuracy_sum_txt += get_AP(k_nearest_txt, train_label, i, K)

		KNN_img[i] = k_nearest_img
		KNN_txt[i] = k_nearest_txt
	print(accuracy_sum_img / train_size)
	print(accuracy_sum_txt / train_size)

	return KNN_img, KNN_txt


def test():	
	discriminator_param = pickle.load(open(DIS_MODEL_BEST_I2I_FILE, 'rb'))
	discriminator = DIS(IMAGE_DIM, TEXT_DIM, HIDDEN_DIM, OUTPUT_DIM, WEIGHT_DECAY, D_LEARNING_RATE, BETA, GAMMA, loss ='svm', param=discriminator_param)
	
	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	sess.run(tf.initialize_all_variables())
	
	I_db = extract_feature(sess, discriminator, train_img, 'image')
	T_db = extract_feature(sess, discriminator, train_txt, 'text')

	knn_img, knn_txt = get_knn(I_db, T_db)

	pdb.set_trace()
	result_dir = '/home/huhengtong/UKD/data/'
	np.save(result_dir + 'teacher_KNN_img.npy', knn_img)
	np.save(result_dir + 'teacher_KNN_txt.npy', knn_txt)





if __name__ == '__main__':
	test()
	