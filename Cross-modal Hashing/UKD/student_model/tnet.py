import scipy.misc
import scipy.io
from ops import *
from setting import *
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

def txt_net(text_input, dimy, bit):
    txtnet={}
    MultiScal = MultiScaleTxt(text_input)
    W_fc1 = tf.random_normal([1, dimy, 6, 4096], stddev=1.0) * 0.01
    b_fc1 = tf.random_normal([1, 4096], stddev=1.0) * 0.01
    fc1W = tf.Variable(W_fc1)
    fc1b = tf.Variable(b_fc1)
    txtnet['conv1'] = tf.nn.conv2d(MultiScal, fc1W, strides=[1, 1, 1, 1], padding='VALID')

    W1_plus_b1 = tf.nn.bias_add(txtnet['conv1'], tf.squeeze(fc1b))
    txtnet['fc1'] = tf.nn.relu(W1_plus_b1)

    txtnet['norm1'] = tf.nn.local_response_normalization(txtnet['fc1'], depth_radius=2, bias=2.000, alpha=0.0001, beta=0.75)

    W_fc2 = tf.random_normal([1, 1, 4096, bit], stddev=1.0) * 0.01
    b_fc2 = tf.random_normal([1, bit], stddev=1.0) * 0.01
    fc2W = tf.Variable(W_fc2)
    fc2b = tf.Variable(b_fc2)
    txtnet['conv2'] = tf.nn.conv2d(txtnet['norm1'], fc2W, strides=[1, 1, 1, 1], padding='VALID')
    W2_plus_b2 = tf.nn.bias_add(txtnet['conv2'], tf.squeeze(fc2b))
    #relu2 = tf.nn.relu(W2_plus_b2)
    txtnet['hash'] = tf.nn.tanh(W2_plus_b2)
    #txtnet['feature'] = relu2

    return tf.squeeze(txtnet['hash'])

def interp_block(text_input, level):
    shape = [1, 1, 5 * level, 1]
    stride = [1, 1, 5 * level, 1]
    prev_layer = tf.nn.avg_pool(text_input, ksize=shape, strides=stride, padding='VALID')
    W_fc1 = tf.random_normal([1, 1, 1, 1], stddev=1.0) * 0.01
    fc1W = tf.Variable(W_fc1)

    prev_layer = tf.nn.conv2d(prev_layer, fc1W, strides=[1, 1, 1, 1], padding='VALID')
    prev_layer = tf.nn.relu(prev_layer)
    prev_layer = tf.image.resize_images(prev_layer, [1, dimText])
    return prev_layer

def MultiScaleTxt(input):
    interp_block1  = interp_block(input, 10)
    interp_block2  = interp_block(input, 6)
    interp_block3  = interp_block(input, 3)
    interp_block6  = interp_block(input, 2)
    interp_block10 = interp_block(input, 1)
    output = tf.concat([input,
                         interp_block10,
                         interp_block6,
                         interp_block3,
                         interp_block2,
                         interp_block1], axis = -1)
    return output
