# import tensorflow as tf
import pdb
import numpy as np

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

try:
    import cPickle as pickle
except:
    import pickle


class DIS():
    def __init__(self, image_dim, text_dim, hidden_dim, output_dim, weight_decay, learning_rate, beta, gamma, loss='svm', param=None):
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.params = []

        self.image_data = tf.placeholder(tf.float32, shape=[None, self.image_dim], name="image_data")
        self.text_data = tf.placeholder(tf.float32, shape=[None, self.text_dim], name="text_data")
        self.image_pos_data = tf.placeholder(tf.float32, shape=[None, self.image_dim], name="image_pos_data")
        self.text_pos_data = tf.placeholder(tf.float32, shape=[None, self.text_dim], name="text_pos_data")
        self.image_neg_data = tf.placeholder(tf.float32, shape=[None, self.image_dim], name="image_neg_data")
        self.text_neg_data = tf.placeholder(tf.float32, shape=[None, self.text_dim], name="text_neg_data")

        with tf.variable_scope('discriminator'):
            if param == None:
                self.Wq_1 = tf.get_variable('Wq_1', [self.image_dim, self.hidden_dim],
                                            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                self.Wq_2 = tf.get_variable('Wq_2', [self.hidden_dim, output_dim],
                                            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                self.Bq_1 = tf.get_variable('Bq_1', [self.hidden_dim], initializer=tf.constant_initializer(0.0))
                
                self.Bq_2 = tf.get_variable('Bq_2', [output_dim], initializer=tf.constant_initializer(0.0))
                self.Wc_1 = tf.get_variable('Wc_1', [self.text_dim, self.hidden_dim],
                                            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                self.Wc_2 = tf.get_variable('Wc_2', [self.hidden_dim, output_dim],
                                            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
                self.Bc_1 = tf.get_variable('Bc_1', [self.hidden_dim], initializer=tf.constant_initializer(0.0))
                self.Bc_2 = tf.get_variable('Bc_2', [output_dim], initializer=tf.constant_initializer(0.0))

            else:
                self.Wq_1 = tf.Variable(param[0])
                self.Wq_2 = tf.Variable(param[1])
                self.Bq_1 = tf.Variable(param[2])
                self.Bq_2 = tf.Variable(param[3])
                
                self.Wc_1 = tf.Variable(param[4])
                self.Wc_2 = tf.Variable(param[5])
                self.Bc_1 = tf.Variable(param[6])
                self.Bc_2 = tf.Variable(param[7])

            self.params.append(self.Wq_1)
            self.params.append(self.Wq_2)
            self.params.append(self.Bq_1)
            self.params.append(self.Bq_2)
            
            self.params.append(self.Wc_1)
            self.params.append(self.Wc_2)
            self.params.append(self.Bc_1)
            self.params.append(self.Bc_2)

        # Given batch query-url pairs, calculate the matching score
        ####compute image_data represent ########
        self.image_rep = tf.nn.xw_plus_b(
            tf.nn.tanh(tf.nn.xw_plus_b(self.image_data, self.Wq_1, self.Bq_1)), self.Wq_2, self.Bq_2)
        self.image_sig = tf.sigmoid(self.image_rep)
        self.image_hash = tf.cast(self.image_sig + 0.5, tf.int32)

        ####compute text_data represent ########
        self.text_rep = tf.nn.xw_plus_b(
            tf.nn.tanh(tf.nn.xw_plus_b(self.text_data, self.Wc_1, self.Bc_1)), self.Wc_2, self.Bc_2)
        self.text_sig = tf.sigmoid(self.text_rep)
        self.text_hash = tf.cast(self.text_sig + 0.5, tf.int32)

        ####compute image_pos represent ########
        self.image_pos_rep = tf.nn.xw_plus_b(
            tf.nn.tanh(tf.nn.xw_plus_b(self.image_pos_data, self.Wq_1, self.Bq_1)), self.Wq_2, self.Bq_2)
        self.image_pos_sig = tf.sigmoid(self.image_pos_rep)
        
        ####compute text_pos represent ########
        self.text_pos_rep = tf.nn.xw_plus_b(
            tf.nn.tanh(tf.nn.xw_plus_b(self.text_pos_data, self.Wc_1, self.Bc_1)), self.Wc_2, self.Bc_2)
        self.text_pos_sig = tf.sigmoid(self.text_pos_rep)

        ####compute image_neg represent ########
        self.image_neg_rep = tf.nn.xw_plus_b(
            tf.nn.tanh(tf.nn.xw_plus_b(self.image_neg_data, self.Wq_1, self.Bq_1)), self.Wq_2, self.Bq_2)
        self.image_neg_sig = tf.sigmoid(self.image_neg_rep)
        
        ####compute text_neg represent ########
        self.text_neg_rep = tf.nn.xw_plus_b(
            tf.nn.tanh(tf.nn.xw_plus_b(self.text_neg_data, self.Wc_1, self.Bc_1)), self.Wc_2, self.Bc_2)
        self.text_neg_sig = tf.sigmoid(self.text_neg_rep)
        
        ###### compute distance #######
        self.pred_distance = tf.reduce_sum(tf.square(self.image_sig - self.text_sig), 1)
        self.pred_img_pos_distance = tf.reduce_sum(tf.square(self.image_sig - self.image_pos_sig), 1) 
        self.pred_txt_pos_distance = tf.reduce_sum(tf.square(self.text_sig - self.text_pos_sig), 1)

        self.pred_i2t_neg_distance = tf.reduce_sum(tf.square(self.image_sig - self.text_neg_sig), 1)
        self.pred_t2i_neg_distance = tf.reduce_sum(tf.square(self.text_sig - self.image_neg_sig), 1)
        self.pred_i2i_neg_distance = tf.reduce_sum(tf.square(self.image_sig - self.image_neg_sig), 1)
        self.pred_t2t_neg_distance = tf.reduce_sum(tf.square(self.text_sig - self.text_neg_sig), 1)

        if loss == 'svm':
            with tf.name_scope('svm_loss'):
                ##### the i2t loss ######## 
                self.i2t_loss = tf.reduce_mean(
                    tf.maximum(0.0, beta + self.pred_distance - self.pred_i2t_neg_distance)) + \
                                self.weight_decay * (tf.nn.l2_loss(self.Wq_1) + tf.nn.l2_loss(self.Wq_2)
                                                     + tf.nn.l2_loss(self.Bq_1) + tf.nn.l2_loss(self.Bq_2)
                                                     + tf.nn.l2_loss(self.Wc_1) + tf.nn.l2_loss(self.Wc_2)
                                                     + tf.nn.l2_loss(self.Bc_1) + tf.nn.l2_loss(self.Bc_2))
                self.i2t_reward = tf.sigmoid(tf.maximum(0.0, beta + self.pred_distance - self.pred_i2t_neg_distance))
                ##### the t2i loss ######## 
                self.t2i_loss = tf.reduce_mean(
                    tf.maximum(0.0, beta + self.pred_distance - self.pred_t2i_neg_distance)) + \
                                self.weight_decay * (tf.nn.l2_loss(self.Wq_1) + tf.nn.l2_loss(self.Wq_2)
                                                     + tf.nn.l2_loss(self.Bq_1) + tf.nn.l2_loss(self.Bq_2)
                                                     + tf.nn.l2_loss(self.Wc_1) + tf.nn.l2_loss(self.Wc_2)
                                                     + tf.nn.l2_loss(self.Bc_1) + tf.nn.l2_loss(self.Bc_2))
                self.t2i_reward = tf.sigmoid(tf.maximum(0.0, beta + self.pred_distance - self.pred_t2i_neg_distance))
                ##### the i2i loss ########
                self.i2i_loss = gamma*tf.reduce_mean(
                    tf.maximum(0.0, beta + self.pred_img_pos_distance - self.pred_i2i_neg_distance)) + \
                                self.weight_decay * (tf.nn.l2_loss(self.Wq_1) + tf.nn.l2_loss(self.Wq_2)
                                                 + tf.nn.l2_loss(self.Bq_1) + tf.nn.l2_loss(self.Bq_2))
                ##### the t2t loss ######## 
                self.t2t_loss = gamma*tf.reduce_mean(
                    tf.maximum(0.0, beta + self.pred_txt_pos_distance - self.pred_t2t_neg_distance)) + \
                                self.weight_decay * (tf.nn.l2_loss(self.Wc_1) + tf.nn.l2_loss(self.Wc_2)
                                                 + tf.nn.l2_loss(self.Bc_1) + tf.nn.l2_loss(self.Bc_2))
                
        global_step = tf.Variable(0, trainable=False)
        lr_step = tf.train.exponential_decay(self.learning_rate, global_step, 20000, 0.9, staircase=True)
        
        self.i2t_optimizer = tf.train.GradientDescentOptimizer(lr_step)
        self.i2t_updates = self.i2t_optimizer.minimize(self.i2t_loss, var_list=self.params)

        self.t2i_optimizer = tf.train.GradientDescentOptimizer(lr_step)
        self.t2i_updates = self.t2i_optimizer.minimize(self.t2i_loss, var_list=self.params)
        
        self.i2i_optimizer = tf.train.GradientDescentOptimizer(lr_step)
        self.i2i_updates = self.i2i_optimizer.minimize(self.i2i_loss, var_list=self.params)
        
        self.t2t_optimizer = tf.train.GradientDescentOptimizer(lr_step)
        self.t2t_updates = self.t2i_optimizer.minimize(self.t2t_loss, var_list=self.params)

    def save_model(self, sess, filename):
        param = sess.run(self.params)
        pickle.dump(param, open(filename, 'wb'))