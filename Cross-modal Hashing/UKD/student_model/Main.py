import os
# import tensorflow as tf
from setting import *
from student_train import SSAH
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

os.environ['CUDA_VISIBLE_DEVICES'] = select_gpu
gpuconfig = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction))

# if not os.path.exists(checkpoint_dir):
#     os.makedirs(checkpoint_dir)
# if not os.path.exists(Savecode):
#     os.makedirs(Savecode)

with tf.Session(config=gpuconfig) as sess:
    model = SSAH(sess)
    model.train() if phase == 'train' else model.test('test')
