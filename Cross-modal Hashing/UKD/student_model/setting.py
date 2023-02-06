import numpy as np
import scipy.io
import argparse
import pdb
# from read_data import read_mirflickr
from load_data import loading_data

def get_dict_values(dictionary, keys):
    num_sam = len(keys)
    values = []
    # print(dictionary.item().keys())
    # print(keys)
    for i in range(num_sam):
        values.append(dictionary.item()[keys[i]])
    return np.array(values)

parser = argparse.ArgumentParser(description='Student Model Training')
parser.add_argument('--gpu', default='0', type=str, help='assign which gpu to use')
parser.add_argument('--bit', default=16, type=int, help='the bit number assigned')
args = parser.parse_args()

# environmental setting: setting the following parameters based on your experimental environment.
select_gpu = args.gpu
per_process_gpu_memory_fraction = 0.9

# Initialize data loader
MODEL_DIR = '/data/WangBoWen/PreTrain/vgg/vgg19.npy'
dataset_dir = 'flickr'
phase = 'train'
checkpoint_dir = './checkpoint'

#SEMANTIC_EMBED = 512
#MAX_ITER = 100
# num_train = 18015
batch_size = 256
image_size = 224
list_dir = '/home/WangBoWen/HashingRetrieval/pythonProject/UKD/data/mir/'

#images, tags, labels = loading_data(DATA_DIR)
#texts_data, labels_data, image_names = read_mirflickr()

X, Y, L = loading_data('flickr')

# retrieval_text_dict = np.load(list_dir + 'retrieval_tags_dict.npy', allow_pickle=True)
# test_text_dict = np.load(list_dir + 'test_tags_dict.npy', allow_pickle=True)
# train_text_dict = np.load(list_dir + 'train_tags_dict.npy', allow_pickle=True)
#
#
# retrieval_label_dict = np.load(list_dir + 'retrieval_labels_dict.npy', allow_pickle=True)
# test_label_dict = np.load(list_dir + 'test_labels_dict.npy', allow_pickle=True)
# train_label_dict = np.load(list_dir + 'train_labels_dict.npy', allow_pickle=True)
#
# retrieval_imgs_dict = np.load(list_dir + 'retrieval_imgs_dict.npy', allow_pickle=True)
# test_imgs_dict = np.load(list_dir + 'test_imgs_dict.npy', allow_pickle=True)
# train_imgs_dict = np.load(list_dir + 'train_imgs_dict.npy', allow_pickle=True)
#
# retrieval_img_path_list = np.load(list_dir + 'retrieval_img_path_list.npy', allow_pickle=True)
# test_img_path_list = np.load(list_dir + 'test_img_path_list.npy', allow_pickle=True)
# train_img_path_list = np.load(list_dir + 'train_img_path_list.npy', allow_pickle=True)
#
#
# retrieval_x = get_dict_values(retrieval_imgs_dict, retrieval_img_path_list)
# query_x = get_dict_values(test_imgs_dict, test_img_path_list)
# train_x = get_dict_values(train_imgs_dict, train_img_path_list)
#
# retrieval_y = get_dict_values(retrieval_text_dict, retrieval_img_path_list)
# query_y = get_dict_values(test_text_dict, test_img_path_list)
# train_y = get_dict_values(train_text_dict, train_img_path_list)
#
# retrieval_label = get_dict_values(retrieval_label_dict, retrieval_img_path_list)
# test_label = get_dict_values(test_label_dict, test_img_path_list)
# train_label = get_dict_values(train_label_dict, train_img_path_list)

retrieval_x = X['retrieval']
query_x = X['query']
train_x = X['train']

retrieval_y = Y['retrieval']
query_y = Y['query']
train_y = Y['train']

retrieval_label = L['retrieval']
test_label = L['query']
train_label = L['train']

# train_L = retrieval_label[0:15000]
train_L = train_label

num_train = train_x.shape[0]
numClass = test_label.shape[1]
dimText = retrieval_y.shape[1]
# dimTxt = dimText
dimLab = test_label.shape[1]


#Sim = (np.dot(train_L, train_L.transpose()) > 0).astype(int)*0.999
teacher_knn_img = np.load('/home/WangBoWen/HashingRetrieval/pythonProject/UKD/data/KNN_db/teacher_KNN_img_t.npy')
teacher_knn_text = np.load('/home/WangBoWen/HashingRetrieval/pythonProject/UKD/data/KNN_db/teacher_KNN_txt_t.npy')
Sim_label = (np.dot(train_label, train_label.transpose()) > 0).astype(np.int32)

Sim = np.zeros((num_train, num_train))
#ind_img = teacher_knn_img[:, 0:8000].astype(np.int32)
#ind_txt = teacher_knn_txt.astype(np.int32)
#print(ind_img.shape)
# pdb.set_trace()
#ap = 0
for i in range(num_train):
    ind = np.concatenate((teacher_knn_img[i], teacher_knn_text[i])).astype(np.int32)
    Sim[i][ind] = 0.999


Epoch = 30

save_freq = 5

bit = args.bit
alpha = 1
gamma = 1
beta = 1
eta = 1
delta = 1

