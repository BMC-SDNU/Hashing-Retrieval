import linecache, pdb

import h5py
import numpy as np
paths = {
    'flickr': '/data/WangBoWen/HashingRetrieval/feature_data/mir_cnn.mat',
    'nuswide': '../Data/raw_nus.mat',
    'coco': '../Data/raw_coco.mat'
}
dataset = h5py.File(paths['flickr'], 'r')

def push_query(query, url, dict):
    if query in dict:
        dict[query].append(url)
    else:
        dict[query] = [url]
    return dict


def make_train_dict(query_list, url_list, label_dim):
    query_url = {}
    query_pos = {}
    query_neg = {}
    query_num = len(query_list)
    url_num = len(url_list)

    for i in range(query_num):
        query = query_list[i]
        for j in range(url_num):
            url = url_list[j]
            if i == j:
                push_query(query, url, query_url)
                push_query(query, url, query_pos)
            else:
                push_query(query, url, query_url)
                push_query(query, url, query_neg)
        if (i+1)%5000==0:
            print(str(i)+'\n')

    return query_url, query_pos, query_neg


def make_test_dict(query_list, url_list, query_label, url_label, label_dim):
    query_url = {}
    query_pos = {}
    query_num = len(query_list)
    url_num = len(url_list)

    for i in range(query_num):
        query = query_list[i]
        for j in range(url_num):
            url = url_list[j]
            #if is_same_cate(query_label[i], url_label[j], label_dim):
            if np.dot(query_label[i], url_label[j]) > 0:
                push_query(query, url, query_url)
                push_query(query, url, query_pos)
            else:
                push_query(query, url, query_url)
        if (i + 1) % 1000 == 0:
            print(str(i) + '\n')
    return query_url, query_pos


def get_dict_values(dictionary, keys):
    num_sam = len(keys)
    values = []
    for i in range(num_sam):
        values.append(dictionary.item()[keys[i]])
    return np.array(values)


def load_all_query_url(list_dir=None, label_dim=None):
    # retrieval_img_dict = np.load(list_dir + 'train_imgs_dict.npy')
    # test_img_dict = np.load(list_dir + 'test_imgs_dict.npy')
    #
    # retrieval_txt_dict = np.load(list_dir + 'train_tags_dict.npy')
    # test_txt_dict = np.load(list_dir + 'test_tags_dict.npy')
    #
    # retrieval_label_dict = np.load(list_dir + 'train_labels_dict.npy')
    # test_label_dict = np.load(list_dir + 'test_labels_dict.npy')
    #
    # retrieval_img_path_list = np.load(list_dir + 'train_img_path_list.npy')
    # test_img_path_list = np.load(list_dir + 'test_img_path_list.npy')
    #
    # # feature
    # retrieval_img = get_dict_values(retrieval_img_dict, retrieval_img_path_list)  # n_tr×4096 array
    # test_img = get_dict_values(test_img_dict, test_img_path_list)  # n_te×4096 array
    # retrieval_txt = get_dict_values(retrieval_txt_dict, retrieval_img_path_list)  # n_tr×1386 array
    # test_txt = get_dict_values(test_txt_dict, test_img_path_list)  # n_te×1386 array
    # retrieval_label = get_dict_values(retrieval_label_dict, retrieval_img_path_list)  # n_tr×24 array
    # test_label = get_dict_values(test_label_dict, test_img_path_list)  # n_te×24 array

    retrieval_img = dataset['I_db'][:].T
    test_img = dataset['I_te'][:].T
    retrieval_txt = dataset['T_db'][:].T
    test_txt = dataset['T_te'][:].T
    retrieval_label = dataset['L_db'][:].T
    test_label = dataset['L_te'][:].T

    
    train_img = retrieval_img[0:15000]
    train_txt = retrieval_txt[0:15000]
    train_label = retrieval_label[0:15000]

    retrieval_img_list = np.arange(retrieval_img.shape[0])   # [0,...,n_tr]
    train_img_list = retrieval_img_list[0:15000]    # [0,...,14999]
    test_img_list = np.arange(retrieval_img_list[-1] + 1, retrieval_img_list[-1] + 1 + test_img.shape[0])  # [n_tr+1,...,n_tr+1+n_te-1]
    retrieval_txt_list = np.arange(test_img_list[-1] + 1, test_img_list[-1] + 1 + retrieval_txt.shape[0])  # [n_tr+n_te+1,...,n_tr+n_te+n_tr]
    train_txt_list = retrieval_txt_list[0:15000]   # [n_tr+n_te+1,...,n_tr+n_te+15000]
    test_txt_list = np.arange(retrieval_txt_list[-1] + 1, retrieval_txt_list[-1] + 1 + test_txt.shape[0])  # [n_tr+n_te+n_tr+1, n_tr+n_te+n_tr+15000]

    # retrieval_img_list = np.arange(retrieval_img.shape[0])  # 0,...,n_db-1
    # # train_img_list = retrieval_img_list[0:15000]
    # train_img_list = np.arange(retrieval_img_list[-1] + 1, retrieval_img_list[-1] + 1 + train_img.shape[0])
    # test_img_list = np.arange(train_img_list[-1] + 1, train_img_list[-1] + 1 + test_img.shape[0])  #
    # retrieval_txt_list = np.arange(test_img_list[-1] + 1, test_img_list[-1] + 1 + retrieval_txt.shape[0])
    # # train_txt_list = retrieval_txt_list[0:15000]
    # train_txt_list = np.arange(retrieval_txt_list[-1] + 1, retrieval_txt_list[-1] + 1 + train_txt.shape[0])
    # test_txt_list = np.arange(train_txt_list[-1] + 1, train_txt_list[-1] + 1 + test_txt.shape[0])


    train_i2t, train_i2t_pos, train_i2t_neg = make_train_dict(train_img_list, train_txt_list, label_dim)
    train_t2i, train_t2i_pos, train_t2i_neg = make_train_dict(train_txt_list, train_img_list, label_dim)

    test_i2t, test_i2t_pos = make_test_dict(test_img_list, retrieval_txt_list, test_label, retrieval_label, label_dim)
    test_t2i, test_t2i_pos = make_test_dict(test_txt_list, retrieval_img_list, test_label, retrieval_label, label_dim)

    return train_i2t, train_i2t_pos, train_i2t_neg, train_t2i, train_t2i_pos, train_t2i_neg, test_i2t, test_i2t_pos, test_t2i, test_t2i_pos


def load_all_feature(list_dir=None):
    # retrieval_img_dict = np.load(list_dir + 'train_imgs_dict.npy')
    # test_img_dict = np.load(list_dir + 'test_imgs_dict.npy')
    #
    # retrieval_txt_dict = np.load(list_dir + 'train_tags_dict.npy')
    # test_txt_dict = np.load(list_dir + 'test_tags_dict.npy')
    #
    # retrieval_img_path_list = np.load(list_dir + 'train_img_path_list.npy')
    # test_img_path_list = np.load(list_dir + 'test_img_path_list.npy')
    #
    # # feature
    # retrieval_img = get_dict_values(retrieval_img_dict, retrieval_img_path_list)
    # test_img = get_dict_values(test_img_dict, test_img_path_list)
    # retrieval_txt = get_dict_values(retrieval_txt_dict, retrieval_img_path_list)
    # test_txt = get_dict_values(test_txt_dict, test_img_path_list)

    retrieval_img = dataset['I_db'][:].T
    test_img = dataset['I_te'][:].T
    retrieval_txt = dataset['T_db'][:].T
    test_txt = dataset['T_te'][:].T

    retrieval_img_list = np.arange(retrieval_img.shape[0])  # [0,n_tr-1]
    test_img_list = np.arange(retrieval_img_list[-1] + 1, retrieval_img_list[-1] + 1 + test_img.shape[0])  # [n_tr, n_tr+n_te-1]
    retrieval_txt_list = np.arange(test_img_list[-1] + 1, test_img_list[-1] + 1 + retrieval_txt.shape[0])  # [n_tr+n_te,n_tr+n_te+n_tr-1]
    test_txt_list = np.arange(retrieval_txt_list[-1] + 1, retrieval_txt_list[-1] + 1 + test_txt.shape[0])  # [n_tr+n_te+n_tr,n_tr+n_te+n_tr+n_te]

    feature_retrieval_img_dict = dict(zip(retrieval_img_list, retrieval_img))
    feature_test_img_dict = dict(zip(test_img_list, test_img))
    feature_retrieval_txt_dict = dict(zip(retrieval_txt_list, retrieval_txt))
    feature_test_txt_dict = dict(zip(test_txt_list, test_txt))

    feature_dict = {**feature_retrieval_img_dict, **feature_test_img_dict}
    feature_dict = {**feature_dict, **feature_retrieval_txt_dict}
    feature_dict = {**feature_dict, **feature_test_txt_dict}
    return feature_dict  # 2×(n_tr+n_te)个 ['0':value,...,'2×(n_tr+n_te)-1':value]


def load_all_label(list_dir=None):
    # retrieval_label_dict = np.load(list_dir + 'train_labels_dict.npy')
    # test_label_dict = np.load(list_dir + 'test_labels_dict.npy')
    #
    # retrieval_img_path_list = np.load(list_dir + 'train_img_path_list.npy')
    # test_img_path_list = np.load(list_dir + 'test_img_path_list.npy')
    #
	# # label
    # retrieval_label = get_dict_values(retrieval_label_dict, retrieval_img_path_list)
    # test_label = get_dict_values(test_label_dict, test_img_path_list)

    retrieval_label = dataset['L_db'][:].T
    test_label = dataset['L_te'][:].T

    retrieval_label_list = np.arange(retrieval_label.shape[0])
    test_label_list = np.arange(retrieval_label_list[-1] + 1, retrieval_label_list[-1] + 1 + test_label.shape[0])

    retrieval_img_label_dict = dict(zip(retrieval_label_list, retrieval_label))
    test_img_label_dict = dict(zip(test_label_list, test_label))
    img_label_dict = {**retrieval_img_label_dict, **test_img_label_dict}
    retrieval_txt_label_dict = dict(zip(retrieval_label_list + len(img_label_dict), retrieval_label))
    test_txt_label_dict = dict(zip(test_label_list + len(img_label_dict), test_label))
    txt_label_dict = {**retrieval_txt_label_dict, **test_txt_label_dict}
    label_dict = {**img_label_dict, **txt_label_dict}
    #print(len(label_dict))
    return label_dict

# if __name__ == '__main__':
#     print(1111)
#     train_i2t, train_i2t_pos, train_i2t_neg, train_t2i, train_t2i_pos, train_t2i_neg, test_i2t, test_i2t_pos, test_t2i, test_t2i_pos = load_all_query_url()
#     feature_dict = load_all_feature()
#     label_dict = load_all_label()
#     pass
