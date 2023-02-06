from setting import *
from tnet import *
# import tensorflow as tf
from ops import *
from calc_hammingranking import calc_map
import os
import pdb
import scipy.io as sio
#from tqdm import tqdm
from read_data import read_image
from vgg19 import Vgg19
#from keras.utils.training_utils import multi_gpu_model

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

class SSAH(object):
    def __init__(self, sess):
        self.train_L = train_L
        self.train_X = train_x
        self.train_Y = train_y

        self.query_L = test_label
        self.query_X = query_x
        self.query_Y = query_y

        self.retrieval_L = retrieval_label
        self.retrieval_X = retrieval_x
        self.retrieval_Y = retrieval_y 

        self.Sim = Sim
        self.img_net = Vgg19
        self.txt_net = txt_net

        self.mse_loss = mse_criterion
        self.sce_loss = sce_criterion

        self.image_size = image_size
        #self.numClass = numClass
        self.dimText = dimText
        #self.dimLab = dimLab
        self.phase = phase
        self.checkpoint_dir = checkpoint_dir
        self.dataset_dir = dataset_dir
        self.bit = bit
        self.num_train = num_train
        self.batch_size = batch_size
        #self.SEMANTIC_EMBED = SEMANTIC_EMBED
        self.sess = sess
        self.build_model()
        self.saver = tf.train.Saver()
        #self.sess = sess

    def build_model(self):
        self.ph = {}
        #self.ph['label_input'] = tf.placeholder(tf.float32, (None, 1, self.numClass, 1), name='label_input')
        self.ph['image_input'] = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='image_input')
        self.ph['text_input'] = tf.placeholder(tf.float32, [None, 1, self.dimText, 1], name='text_input')
        self.ph['lr_hash'] = tf.placeholder('float32', (), name='lr_hash')
        #self.ph['lr_lab'] = tf.placeholder('float32', (), name='lr_lab')
        self.ph['lr_img'] = tf.placeholder('float32', (), name='lr_img')
        self.ph['lr_txt'] = tf.placeholder('float32', (), name='lr_txt')
        #self.ph['lr_dis'] = tf.placeholder('float32', (), name='lr_discriminator')
        self.ph['keep_prob'] = tf.placeholder('float32', (), name='keep_prob')
        self.ph['Sim'] = tf.placeholder('float32', [self.num_train, self.batch_size], name='Sim')
        self.ph['F'] = tf.placeholder('float32', [None, self.bit], name='F')
        self.ph['G'] = tf.placeholder('float32', [None, self.bit], name='G')
        #self.ph['H'] = tf.placeholder('float32', [None, self.bit], name='H')
        #self.ph['L_batch'] = tf.placeholder('float32', [None, self.numClass], name='L_batch')
        self.ph['B_batch'] = tf.placeholder('float32', [None, self.bit], name='b_batch')
        #self.ph['dropout'] = tf.placeholder('float32', [None], name='dropout')

        # construct image network
        #self.Hsh_I = self.img_net(self.ph['image_input'], self.bit, self.numClass)
        #self.Img_model = self.img_net(self.ph['image_input'], 1, ['fc8'], self.bit, modelPath=MODEL_DIR)
        self.Img_model = self.img_net(vgg19_npy_path=MODEL_DIR, dropout=self.ph['keep_prob'], bit=self.bit)
        self.Img_model.build(self.ph['image_input'])
        self.Hsh_I = self.Img_model.fc_hash

        # construct text network
        #with tf.device('/cpu:1'):
        self.Hsh_T = self.txt_net(self.ph['text_input'], self.dimText, self.bit)

        # train img_net combined with lab_net
        # theta_I_1 = 1.0 / 2 * tf.matmul(self.ph['L_fea'], tf.transpose(self.Fea_I))
        # Loss_pair_Fea_I = self.mse_loss(tf.multiply(self.ph['Sim'], theta_I_1), tf.log(1.0 + tf.exp(theta_I_1)))
        theta_I_2 = 1.0 / 2 * tf.matmul(self.ph['G'], tf.transpose(self.Hsh_I))
        Loss_pair_Hsh_I = self.mse_loss(tf.multiply(self.ph['Sim'], theta_I_2), tf.log(1.0 + tf.exp(theta_I_2)))
        Loss_quant_I = self.mse_loss(self.ph['B_batch'], self.Hsh_I)
        # Loss_label_I = self.mse_loss(self.ph['L_batch'], self.Lab_I)
        # Loss_adver_I = self.sce_loss(logits=self.isfrom_IL, labels=tf.ones_like(self.isfrom_IL))
        self.loss_i = gamma * Loss_pair_Hsh_I + beta * Loss_quant_I

            # train txt_net combined with lab_net
        # theta_T_1 = 1.0 / 2 * tf.matmul(self.ph['L_fea'], tf.transpose(self.Fea_T))
        # Loss_pair_Fea_T = self.mse_loss(tf.multiply(self.ph['Sim'], theta_T_1), tf.log(1.0 + tf.exp(theta_T_1)))
        theta_T_2 = 1.0 / 2 * tf.matmul(self.ph['F'], tf.transpose(self.Hsh_T))
        Loss_pair_Hsh_T = self.mse_loss(tf.multiply(self.ph['Sim'], theta_T_2), tf.log(1.0 + tf.exp(theta_T_2)))
        Loss_quant_T = self.mse_loss(self.ph['B_batch'], self.Hsh_T)
        # Loss_label_T = self.mse_loss(self.ph['L_batch'], self.Lab_T)
        # Loss_adver_T = self.sce_loss(logits=self.isfrom_TL, labels=tf.ones_like(self.isfrom_TL))
        self.loss_t = gamma * Loss_pair_Hsh_T + beta * Loss_quant_T

    def train(self):
        # """Train"""
        # for gpu_id in range(2):
        #     with tf.device('/gpu:%d' % gpu_id):
        optimizer = tf.train.AdamOptimizer(self.ph['lr_hash'])

        gradient_i = optimizer.compute_gradients(self.loss_i)
        self.train_img = optimizer.apply_gradients(gradient_i)

        gradient_t = optimizer.compute_gradients(self.loss_t)
        self.train_txt = optimizer.apply_gradients(gradient_t)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        #self.Img_model.loadModel(self.sess)

        var = {}
        var['batch_size'] = batch_size
        var['F'] = np.random.randn(self.num_train, self.bit)
        var['G'] = np.random.randn(self.num_train, self.bit)
        #var['H'] = np.random.randn(self.num_train, self.bit)

        var['B'] = np.sign(var['G'] + var['F'])

        # Iterations
        for epoch in range(Epoch):
            print('++++++++Starting Training ++++++++')
            lr_img = 0.0001
            lr_txt = 0.0001

            for iter in range(2):
                var['F'] = self.train_img_net(var, lr_img)
                var['G'] = self.train_txt_net(var, lr_txt)

                # B_i = np.sign(var['F'])
                # B_t = np.sign(var['G'])
                var['B'] = np.sign(var['G'] + var['F'])

                train_loss = self.calc_loss(var['B'], var['F'], var['G'], Sim, alpha, beta)
                #train_txtNet_loss = self.calc_loss(B_t, var['F'], var['G'], Sim, alpha, beta)

                print('---------------------------------------------------------------')
                print('...epoch: %3d, loss: %3.3f' % (epoch, train_loss))
                print('---------------------------------------------------------------')

                #var['B'] = np.sign(var['G'] + var['F'])

            print("********test************")
            self.test(self.phase)
            # 保存模型
            if np.mod(epoch, save_freq) == 0:
                self.save(self.checkpoint_dir, epoch)

    def test(self, phase):
        test = {}
        print('==========================================================')
        print('  ====                 Test map in all              ====')
        print('==========================================================')

        if phase == 'test' and self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        test['qBX'] = self.generate_code(self.query_X, self.bit, "image")
        test['qBY'] = self.generate_code(self.query_Y, self.bit, "text")
        # test['rBX'] = self.generate_code(self.train_X, self.bit, "image")
        # test['rBY'] = self.generate_code(self.train_Y, self.bit, "text")

        test['rBX'] = self.generate_code(self.retrieval_X, self.bit, "image")
        test['rBY'] = self.generate_code(self.retrieval_Y, self.bit, "text")

        test['mapi2t'] = calc_map(test['qBX'], test['rBY'], self.query_L, self.retrieval_L)
        test['mapt2i'] = calc_map(test['qBY'], test['rBX'], self.query_L, self.retrieval_L)
        test['mapi2i'] = calc_map(test['qBX'], test['rBX'], self.query_L, self.retrieval_L)
        test['mapt2t'] = calc_map(test['qBY'], test['rBY'], self.query_L, self.retrieval_L)
        print('==================================================')
        print('...test map: map(i->t): %3.3f, map(t->i): %3.3f' % (test['mapi2t'], test['mapt2i']))
        print('...test map: map(t->t): %3.3f, map(i->i): %3.3f' % (test['mapt2t'], test['mapi2i']))
        print('==================================================')

    
    def train_img_net(self, var, lr_img):
        print('update image_net')
        F = var['F']
        batch_size = var['batch_size']
        num_train = self.train_X.shape[0]
        for iter in range(num_train // batch_size):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            sample_L = self.train_L[ind, :]

            # image_path = self.train_X[ind]
            # image = read_image(image_path).astype(np.float64)
            image = self.train_X[ind,:,:,:]

            # print(image)
            # pdb.set_trace()
            #image = image - self.meanpix.astype(np.float64)
            #S = calc_neighbor(self.train_L, sample_L)
            S = Sim[ind].transpose()
            Hsh_I = self.sess.run(self.Hsh_I, feed_dict={self.ph['image_input']: image, self.ph['keep_prob']: 0.5})
            # print(Hsh_I)
            # pdb.set_trace()

            #Hsh_I = result[0]
            # Fea_I = result[1]
            # Lab_I = result[2]
    
            F[ind, :] = Hsh_I
            # Feat_I[ind, :] = Fea_I
            # LABEL_I[ind, :] = Lab_I

            self.train_img.run(feed_dict={self.ph['Sim']: S,
                                        self.ph['G']: var['G'],
                                        self.ph['B_batch']: np.sign(Hsh_I),
                                        #self.ph['L_batch']: self.train_L[ind, :],
                                        #self.ph['L_fea']: var['feat_L'],
                                        self.ph['lr_hash']: lr_img,
                                        #self.ph['I_fea_batch']: var['feat_I'].reshape([var['feat_I'].shape[0], 1, var['feat_I'].shape[1], 1]),
                                        self.ph['image_input']: image,
                                        self.ph['keep_prob']: 0.5})
        return F

    def train_txt_net(self, var, lr_txt):
        print('update text_net')
    
        G = var['G']
        batch_size = var['batch_size']
        num_train = self.train_Y.shape[0]
        for iter in range(num_train // batch_size):
            index = np.random.permutation(num_train)
            ind = index[0: batch_size]
            sample_L = self.train_L[ind, :]

            text = self.train_Y[ind, :].astype(np.float32)
            text = text.reshape([text.shape[0], 1, text.shape[1], 1])
    
            #S = calc_neighbor(self.train_L, sample_L)
            S = Sim[ind].transpose()
            Hsh_T = self.sess.run(self.Hsh_T, feed_dict={self.ph['text_input']: text})
            #Hsh_T = result[0]
    
            G[ind, :] = Hsh_T

            self.train_txt.run(feed_dict={self.ph['text_input']: text,
                                             self.ph['Sim']: S,
                                             self.ph['F']: var['F'],
                                             self.ph['B_batch']: np.sign(Hsh_T),
                                             #self.ph['L_batch']: self.train_L[ind, :],
                                             #self.ph['L_fea']: var['feat_L'],
                                             self.ph['lr_hash']: lr_txt,
                                             self.ph['keep_prob']: 1.0})
        return G

    def generate_code(self, Modal, bit, generate):
        batch_size = 128
        if generate=="image":
            num_data = len(Modal)
            index = np.linspace(0, num_data - 1, num_data).astype(int)
            B = np.zeros([num_data, bit], dtype=np.float32)
            for iter in range(num_data // batch_size + 1):
                ind = index[iter * batch_size: min((iter + 1) * batch_size, num_data)]
                #mean_pixel = np.repeat(self.meanpix[:, :, :, np.newaxis], len(ind), axis=3)
                # image_path = Modal[ind]
                # image = read_image(image_path).astype(np.float64)
                image = Modal[ind, :, :, :]

                #image = image - mean_pixel.astype(np.float64).transpose(3, 0, 1, 2)
                Hsh_I = self.Hsh_I.eval(feed_dict={self.ph['image_input']: image, self.ph['keep_prob']: 1.0})
                B[ind, :] = Hsh_I
        else:
            num_data = Modal.shape[0]
            index = np.linspace(0, num_data - 1, num_data).astype(int)
            B = np.zeros([num_data, bit], dtype=np.float32)
            for iter in range(num_data // batch_size + 1):
                ind = index[iter * batch_size: min((iter + 1) * batch_size, num_data)]
                text = Modal[ind].astype(np.float32)
                text = text.reshape([text.shape[0], 1, text.shape[1], 1])
                Hsh_T = self.Hsh_T.eval(feed_dict={self.ph['text_input']: text})
                B[ind, :] = Hsh_T
        B = np.sign(B)
        return B
    
    
    def calc_loss(self, B, F, G, Sim, alpha, beta):
        theta = np.matmul(F, np.transpose(G)) / 2
        term1 = np.sum(np.log(1 + np.exp(theta)) - Sim * theta)

        term2 = np.sum(np.power(B-F, 2) + np.power(B-G, 2))
    
        loss = alpha * term1 + beta * term2 #+ gamma * term3 + eta * term4
        return loss


    def calc_isfrom_acc(self, train_isfrom_, Train_ISFROM):
        erro = Train_ISFROM.shape[0] - np.sum(np.equal(np.sign(train_isfrom_ - 0.5), np.sign(Train_ISFROM - 0.5)).astype(int))
        acc = np.divide(np.sum(np.equal(np.sign(train_isfrom_ - 0.5), np.sign(Train_ISFROM - 0.5)).astype('float32')), Train_ISFROM.shape[0])
        return erro, acc


    def save(self, checkpoint_dir, step):
        model_name = "SSAH"
        model_dir = "%s_%s" % (self.dataset_dir, self.bit)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_dir, self.bit)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
