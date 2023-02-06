import numpy as np
import pdb
import utils as ut

def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)

def average_precision(r):
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)

def calcu_map(query_pos_test, query_index_url_test, hash_dict, label_dict):
    rs = []
    map = 0
    # print(len(query_pos_test))
    for query in query_pos_test.keys():
# 		pos_set = set(query_pos_test[query])
        pred_list = query_index_url_test[query]

        pred_list_score = []
        query_hash = hash_dict[query]
        query_L = label_dict[query]
        
        code_length = query_hash.shape[0]

        candidates_hash = []
        retrieval_L = []
        for candidate in pred_list:
#			score = 0
            candidates_hash.append(hash_dict[candidate])
            retrieval_L.append(label_dict[candidate])
        candidates_hash = np.asarray(candidates_hash)
        retrieval_L = np.asarray(retrieval_L)
#		pred_list_score = code_length - np.sum(np.bitwise_xor(query_hash, candidates_hash), axis=1)  
        hamming_dist = np.sum(np.bitwise_xor(query_hash, candidates_hash), axis=1)
        idx = np.argsort(hamming_dist)

        gnd = (np.dot(query_L, retrieval_L.transpose()) > 0).astype(np.float32)
        gnd = gnd[idx]
        tsum = np.sum(gnd)
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map = map + np.mean(count / (np.squeeze(tindex)))
    map = map / len(query_pos_test)
    return map
    
def MAP(sess, model, test_i2t_pos, test_i2t, test_t2i_pos, test_t2i, feature_dict, label_dict):
    hash_dict_img = get_hash_dict(sess, model, feature_dict, 'img')
    hash_dict_txt = get_hash_dict(sess, model, feature_dict, 'txt')
    hash_dict_img.update(hash_dict_txt)
    hash_dict = hash_dict_img
    print(test_i2t_pos.keys())
    print(test_i2t.keys())
    map_i2t = calcu_map(test_i2t_pos, test_i2t, hash_dict, label_dict)
    map_t2i = calcu_map(test_t2i_pos, test_t2i, hash_dict, label_dict)
    return map_i2t, map_t2i

def get_hash_dict(sess, model, feature_dict, flag):
    batch_size = 256
    hash_list = []
    if flag == 'img':
        index = 0
        max_idx = 20015 + 5000
    if flag == 'txt':
        index = 20015 + 5000
        max_idx = 40030 + 10000
    while index < max_idx:
        input_data = []
        if index + batch_size <= max_idx:
            for i in range(index, index+batch_size):
                input_data.append(feature_dict[i])
        else:
            for i in range(index, max_idx):
                input_data.append(feature_dict[i])
        index += batch_size
        input_data = np.asarray(input_data)
        if flag == 'img':
            output_hash = sess.run(model.image_hash, feed_dict={model.image_data: input_data})
        if flag == 'txt':
            output_hash = sess.run(model.text_hash, feed_dict={model.text_data: input_data})
        hash_list.append(output_hash)
    hash_list = np.concatenate(hash_list)
    list_idx = np.arange(max_idx - 20015 - 5000, max_idx)
#	print(list_idx.shape)    
    hash_dict = dict(zip(list_idx, hash_list))
    return hash_dict


def get_labels(list_dir):
    train_label_dict = np.load(list_dir + 'train_labels_dict.npy')
    test_label_dict = np.load(list_dir + 'test_labels_dict.npy')

    train_img_path_list = np.load(list_dir + 'train_img_path_list.npy')
    test_img_path_list = np.load(list_dir + 'test_img_path_list.npy')

    train_label = ut.get_dict_values(train_label_dict, train_img_path_list)
    test_label = ut.get_dict_values(test_label_dict, test_img_path_list)
    return train_label, test_label
