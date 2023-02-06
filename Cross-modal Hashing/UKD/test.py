import numpy as np

from teacher_model.utils_bkp import *
def test(test0, test1, test2):
    cal(test1)
def cal(query):
    for k in query.keys():
        print(k)
    print(len(query))


if __name__ == '__main__':
    print(1111)
    # train_i2t, train_i2t_pos, train_i2t_neg, train_t2i, train_t2i_pos, train_t2i_neg, test_i2t, test_i2t_pos, test_t2i, test_t2i_pos = load_all_query_url()
    # path = '/home/WangBoWen/HashingRetrieval/pythonProject/UKD/dataset/'
    # # test_i2t = np.load(path + 'test_i2t.npy', allow_pickle=True).item()
    # test_i2t_pos = np.load(path + 'test_i2t_pos.npy', allow_pickle=True).item()
    # test_t2i = np.load(path + 'test_t2i.npy', allow_pickle=True).item()
    # # test_t2i_pos = np.load(path + 'train_i2t_pos.npy', allow_pickle=True).item()
    # test(test_i2t_pos,test_i2t_pos,test_t2i)
    # for i in range(10):
    #     # print(i)
    #     if i == 1:
    #         continue
    #     for j in range(100,110):
    #         print('j ',j)
    #     print(i)
    # num_data = 5000
    # batch_size = 2
    # index = np.linspace(0, num_data - 1, num_data).astype(int)
    # ind = index[0 * batch_size: min((0 + 1) * batch_size, num_data)]
    # # print(ind)
    # # print(type(ind))
    # A = [[1, 3, 0],
    #      [3, 2, 0],
    #      [0, 2, 1],
    #      [1, 1, 4],
    #      [3, 2, 2],
    #      [0, 1, 0],
    #      [1, 3, 1],
    #      [0, 4, 1],
    #      [2, 4, 2],
    #      [3, 3, 1]]
    # A = np.array(A)
    # choice = np.random.choice(10, size=2, replace=False)
    # print(np.random.choice(10, size=2, replace=False))
    # print(A[ind,:])

    train_label_list = np.arange(2000)
    test_label_list = np.arange(train_label_list[-1] + 1, train_label_list[-1]+1+np.arange(test_label.shape[0]))
    pass