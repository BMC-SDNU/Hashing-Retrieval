import random, pdb, time

# import tensorflow as tf
import numpy as np
import utils as ut
# from map_old import *
from map import *
from dis_model_nn import DIS
try:
    import cPickle as pickle
except:
    import pickle

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


GPU_ID = 0
OUTPUT_DIM = 128

SELECTNUM = 1
SAMPLERATIO = 20

WHOLE_EPOCH = 1
D_EPOCH = 200
GS_EPOCH = 1000
D_DISPLAY = 1

IMAGE_DIM = 4096
TEXT_DIM = 1386
HIDDEN_DIM = 8192
CLASS_DIM = 24
BATCH_SIZE = 256
WEIGHT_DECAY = 0.01
D_LEARNING_RATE = 0.01
LAMBDA = 0
BETA = OUTPUT_DIM / 8.0
GAMMA = 0.1


list_dir = '/home/WangBoWen/HashingRetrieval/pythonProject/UKD/data/mir/'
pretrain_list = '/home/WangBoWen/HashingRetrieval/pythonProject/UKD/data/pretrain_5000/'
DIS_MODEL_PRETRAIN_FILE = list_dir + 'new_teacher_best_pretrain_' + str(OUTPUT_DIM) + '.model'

train_i2t, train_i2t_pos, train_i2t_neg, train_t2i, train_t2i_pos, train_t2i_neg, test_i2t, test_i2t_pos, test_t2i, test_t2i_pos = ut.load_all_query_url(list_dir, CLASS_DIM)

feature_dict = ut.load_all_feature(list_dir)
label_dict = ut.load_all_label(list_dir)


def generate_samples(train_pos, train_neg, flag):
    data = []
    for query in train_pos:
        pos_list = train_pos[query]
        candidate_neg_list = train_neg[query]

        random.shuffle(pos_list)
        random.shuffle(candidate_neg_list)

        for i in range(SELECTNUM):
            data.append((query, pos_list[i], candidate_neg_list[i]))

    random.shuffle(data)
    return data

def train_discriminator(sess, discriminator, dis_train_list, flag):
    train_size = len(dis_train_list)
    index = 1
    while index < train_size:
        input_query = []
        input_pos = []
        input_neg = []

        if index + BATCH_SIZE <= train_size:
            for i in range(index, index + BATCH_SIZE):
                query, pos, neg = dis_train_list[i]
                input_query.append(feature_dict[query])
                input_pos.append(feature_dict[pos])
                input_neg.append(feature_dict[neg])
        else:
            for i in range(index, train_size):
                query, pos, neg = dis_train_list[i]
                input_query.append(feature_dict[query])
                input_pos.append(feature_dict[pos])
                input_neg.append(feature_dict[neg])

        index += BATCH_SIZE

        query_data = np.asarray(input_query)
        input_pos = np.asarray(input_pos)
        input_neg = np.asarray(input_neg)

        if flag == 'i2t':
            d_loss = sess.run(discriminator.i2t_loss,
                         feed_dict={discriminator.image_data: query_data,
                                    discriminator.text_data: input_pos,
                                    discriminator.text_neg_data: input_neg})
            _ = sess.run(discriminator.i2t_updates,
                         feed_dict={discriminator.image_data: query_data,
                                    discriminator.text_data: input_pos,
                                    discriminator.text_neg_data: input_neg})
        elif flag == 't2i':
            d_loss = sess.run(discriminator.t2i_loss,
                         feed_dict={discriminator.text_data: query_data,
                                    discriminator.image_data: input_pos,
                                    discriminator.image_neg_data: input_neg})
            _ = sess.run(discriminator.t2i_updates,
                         feed_dict={discriminator.text_data: query_data,
                                    discriminator.image_data: input_pos,
                                    discriminator.image_neg_data: input_neg})

    print('D_Loss: %.4f' % d_loss)
    return discriminator

def main():
    with tf.device('/gpu:' + str(GPU_ID)):
        # dis_param = cPickle.load(open(DIS_MODEL_NEWEST_FILE))
        # discriminator = DIS(IMAGE_DIM, TEXT_DIM, HIDDEN_DIM, OUTPUT_DIM, WEIGHT_DECAY, D_LEARNING_RATE, BETA, GAMMA, loss = 'svm', param = dis_param)
        discriminator = DIS(IMAGE_DIM, TEXT_DIM, HIDDEN_DIM, OUTPUT_DIM, WEIGHT_DECAY, D_LEARNING_RATE, BETA, GAMMA, loss = 'svm', param = None)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.initialize_all_variables())

        print('start adversarial training')
        map_best_val_gen = 0.0
        map_best_val_dis = 0.0

        for epoch in range(WHOLE_EPOCH):
            print('Training D ...')
            for d_epoch in range(D_EPOCH):
                print('d_epoch: ' + str(d_epoch))
                if d_epoch % GS_EPOCH == 0:
                    print('negative text sampling for d using g ...')
                    dis_train_i2t_list = generate_samples(train_i2t_pos, train_i2t_neg, 'i2t')
                    print('negative image sampling for d using g ...')
                    dis_train_t2i_list = generate_samples(train_t2i_pos, train_t2i_neg, 't2i')

                discriminator = train_discriminator(sess, discriminator, dis_train_i2t_list, 'i2t')
                discriminator = train_discriminator(sess, discriminator, dis_train_t2i_list, 't2i')

                if (d_epoch + 1) % (D_DISPLAY) == 0:
                    i2t_test_map, t2i_test_map = MAP(sess, discriminator, test_i2t_pos, test_i2t, test_t2i_pos, test_t2i, feature_dict, label_dict)
                    print('---------------------------------------------------------------')
                    print('pretrain_I2T_Test_MAP: %.4f' % i2t_test_map)
                    print('pretrain_T2I_Test_MAP: %.4f' % t2i_test_map)

                    average_map = 0.5 * (i2t_test_map + t2i_test_map)
                    if average_map > map_best_val_dis:
                        map_best_val_dis = average_map
                        discriminator.save_model(sess, DIS_MODEL_PRETRAIN_FILE)

        sess.close()
if __name__ == '__main__':
    main()
