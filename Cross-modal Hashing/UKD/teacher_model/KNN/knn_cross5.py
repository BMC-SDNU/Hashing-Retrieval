import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import pdb


import h5py
import numpy as np
paths = {
    'flickr': '/data/WangBoWen/HashingRetrieval/feature_data/mir_cnn.mat',
    'nuswide': '../Data/raw_nus.mat',
    'coco': '../Data/raw_coco.mat'
}
dataset = h5py.File(paths['flickr'], 'r')

# train_img = dataset['I_tr'][:].T
# train_txt = dataset['T_tr'][:].T
# train_labels = dataset['L_tr'][:].T

train_img = dataset['I_db'][:].T
train_txt = dataset['T_db'][:].T
train_labels = dataset['L_db'][:].T



# dataset = 'mir'
K = 10
# train_size = 18015
train_size = train_labels.shape[0]
label_dim = 24

# train_img = np.load('/....../imgs_train.npy')
# train_txt = np.load('/....../texts_train.npy')
# train_labels = np.load('/....../labels_train.npy')

# def get_feature(feature_string_list, train_size):
# 	feature_list = []
# 	for i in range(train_size):
# 		feature_string = feature_string_list[i].split()
# 		feature_float_list = []
# 		for j in range(len(feature_string)):
# 			feature_float_list.append(float(feature_string[j]))
# 		feature_list.append(feature_float_list)
#
# 	return np.asarray(feature_list)


def get_distance(vec1, vec2):
    return np.sqrt(np.sum(np.square(vec1 - vec2)))
	
def is_same_cate(labelA, labelB, label_dim):
    # labelA = strA.split()
    # labelB = strB.split()
    for i in range(label_dim):
        if labelA[i] == 1 and labelA[i] == labelB[i]:
            return True
    return False
	
def get_AP(k_nearest, label, query_index, k, label_dim):
    score = 0.0
    for i in range(k):
        if np.dot(label[query_index], label[int(k_nearest[i])]) > 0:
        #if is_same_cate(label[query_index], label[int(k_nearest[i])], label_dim):
            score += 1.0
    return score / k
	
	

# dataset_dir = '../' + dataset + '/'
# list_dir = dataset_dir + 'list/'
# feature_dir = dataset_dir + 'feature/'
# result_dir = './' + dataset + '/'

# train_img_string_list = open(feature_dir + 'train_img.txt', 'r').read().split('\n')
# train_txt_string_list = open(feature_dir + 'train_txt.txt', 'r').read().split('\n')
# train_label = open(list_dir + 'train_label.txt', 'r').read().split('\r\n')

# train_img = get_feature(train_img_string_list, train_size)
# train_txt = get_feature(train_txt_string_list, train_size)

distance_img = pdist(train_img, 'euclidean')
distance_txt = pdist(train_txt, 'euclidean')
# np.save(result_dir + 'distance_img.npy', distance_img)
# np.save(result_dir + 'distance_txt.npy', distance_txt)
# distance_img = np.load(result_dir + 'distance_img.npy')
# distance_txt = np.load(result_dir + 'distance_txt.npy')

distance_img = squareform(distance_img)
distance_txt = squareform(distance_txt)
		
KNN_img = np.zeros((train_size, K))
KNN_txt = np.zeros((train_size, K))
KNN_cross = np.zeros((train_size, K))
accuracy_sum_img = 0
accuracy_sum_txt = 0
accuracy_sum_cross = 0
	
for i in range(train_size):
    k_nearest_img = np.argsort(distance_img[i])[0:K]
    k_nearest_txt = np.argsort(distance_txt[i])[0:K]
    k_nearest_cross = np.zeros((K))
    for j in range(K // 2):
        k_nearest_cross[j] = k_nearest_img[j]

    for j in range(K // 2):
        k_nearest_cross[j+K // 2] = k_nearest_txt[j]

    accuracy_sum_img += get_AP(k_nearest_img, train_labels, i, K, label_dim)
    accuracy_sum_txt += get_AP(k_nearest_txt, train_labels, i, K, label_dim)
    accuracy_sum_cross += get_AP(k_nearest_cross, train_labels, i, K, label_dim)

    KNN_img[i] = k_nearest_img
    KNN_txt[i] = k_nearest_txt
    KNN_cross[i] = k_nearest_cross
	
		
print(accuracy_sum_img / train_size)
print(accuracy_sum_txt / train_size)
print(accuracy_sum_cross / train_size)

# pdb.set_trace()
# result_dir = '/home/huhengtong/UKD/data/'
result_dir = '/home/WangBoWen/HashingRetrieval/pythonProject/UKD/data/KNN_db/'
np.save(result_dir + 'KNN_img_db.npy', KNN_img)
np.save(result_dir + 'KNN_txt_db.npy', KNN_txt)
np.save(result_dir + 'KNN_cross5_db.npy', KNN_cross)

