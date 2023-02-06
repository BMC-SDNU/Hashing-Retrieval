import numpy as np

import teacher_model.utils as ut

CLASS_DIM = 24


if __name__ == '__main__':
    list_dir = '/home/WangBoWen/HashingRetrieval/pythonProject/UKD/data/mir/'

    # 加载数据集
    # train_i2t, train_i2t_pos, train_i2t_neg, train_t2i, train_t2i_pos, train_t2i_neg, test_i2t, test_i2t_pos, test_t2i, test_t2i_pos = ut.load_all_query_url(list_dir=list_dir, label_dim=CLASS_DIM)

    feature_dict = ut.load_all_feature(list_dir=list_dir)
    # label_dict = ut.load_all_label(list_dir=list_dir)

    # mir_data_all_query_url = {'train_i2t':train_i2t, 'train_i2t_pos':train_i2t_pos}

    # mir_data_all_query_url = {'train_i2t':train_i2t,'train_i2t_pos':train_i2t_pos,'train_i2t_neg':train_i2t_neg,
    #             'train_t2i':train_t2i,'train_t2i_pos':train_t2i_pos,'train_t2i_neg':train_t2i_neg,
    #             'test_i2t':test_i2t,'test_i2t_pos':test_i2t_pos,
    #             'test_t2i':test_t2i,'test_t2i_pos':test_t2i_pos}

    # mir_data_feature_dict = {'feature_dict':feature_dict}
    #
    # mir_data_label_dict = {'label_dict':label_dict}
    # mir_data_all_query_url = {'da':{'0':[1,2,3,4],'1':[1,2,3,4]}}
    path = '/home/WangBoWen/HashingRetrieval/pythonProject/UKD/dataset/test/'
    # train_i2t = np.load(path+'train_i2t_pos.npy', allow_pickle=True)
    # train_list = np.load(path+'train_i2t.npy', allow_pickle=True)
    # train = train_list.item()
    # print(train[0])
    # print(222)

    print(len(feature_dict))
    # np.save(path+'train_i2t.npy', train_i2t)
    # np.save(path+'train_i2t_pos.npy', train_i2t_pos)
    # np.save(path+'train_i2t_neg.npy', train_i2t_neg)
    # np.save(path+'train_t2i.npy', train_t2i)
    # np.save(path+'train_t2i_pos.npy', train_t2i_pos)
    # np.save(path+'train_t2i_neg.npy', train_t2i_neg)
    # np.save(path+'test_i2t.npy', test_i2t)
    # np.save(path+'test_i2t_pos.npy', test_i2t_pos)
    # np.save(path+'test_t2i.npy', test_t2i)
    # np.save(path+'test_t2i_pos.npy', test_t2i_pos)
    # np.save(path+'mir_data_feature_dict.npy', feature_dict)
    # np.save(path+'mir_data_label_dict.npy', label_dict)



