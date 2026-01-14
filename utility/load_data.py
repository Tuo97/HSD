import pickle
import numpy as np
from time import time
from tqdm import tqdm
import scipy.sparse as sp
import random as rd

class Data(object):
    def __init__(self, args):

        self.path = args.data_path + args.dataset
        self.n_batch = args.n_batch
        self.batch_size = args.batch_size

        self.train_num = args.train_num
        self.sample_num = args.sample_num

        # get number of users and items
        # Beauty dataset
        self.n_users = 22363
        self.n_items = 12101

        # Toys dataset
        # self.n_users = 19412
        # self.n_items = 11924

        # Phones dataset
        # self.n_users = 27879
        # self.n_items = 10429

        self.n_train, self.n_test = 0, 0

        self.exist_users = []
        self.train_items, self.test_set = {}, {}
        ############################################################################ 数据集划分
        with open("./data/AmazonBeauty/train.pickle", "rb") as f:
            self.train_items = pickle.load(f)
        with open("./data/AmazonBeauty/test.pickle", "rb") as f:
            self.test_set = pickle.load(f)

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)  # 创建稀疏评分图

        for u_id in range(self.n_users):
            self.exist_users.append(u_id)
            train_items = self.train_items[u_id]
            self.n_train += len(train_items)
            test_items = self.test_set[u_id]
            self.n_test += len(test_items)
            for i_id in train_items:
                self.R[u_id, i_id] = 1.

        self.print_statistics()

    def get_adj_mat(self):
        adj_mat = self.create_adj_mat()

        def normalized_adj_double(adj):
            adj_mat = adj
            rowsum = np.array(adj_mat.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()  # 生成D^-1/2度矩阵
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)  # 双边标准化
            print('generate pre adjacency matrix.')
            pre_adj_mat = norm_adj.tocsr()
            return pre_adj_mat

        norm_adj = normalized_adj_double(adj_mat)

        return adj_mat, norm_adj

    def create_adj_mat(self):
        t1 = time()
        rows = self.R.tocoo().row
        cols = self.R.tocoo().col
        new_rows = np.concatenate([rows, cols + self.n_users], axis=0)
        new_cols = np.concatenate([cols + self.n_users, rows], axis=0)
        adj_mat = sp.coo_matrix((np.ones(len(new_rows)), (new_rows, new_cols)), shape=[self.n_users + self.n_items, self.n_users + self.n_items]).tocsr().tocoo()
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)
        return adj_mat.tocsr()

    def sample_NUS(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)                      #抽取用户的batch_size
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        users = np.array(users)
        pos_items = np.array(pos_items)
        neg_items = np.array(neg_items)

        return users, pos_items, neg_items

    def mini_batch(self, batch_idx):
        st = batch_idx * self.batch_size
        ed = min((batch_idx + 1) * self.batch_size, len(self.train_data))
        batch_data = self.train_data[st: ed]
        users = batch_data[:, 0]
        pos_items = batch_data[:, 1]
        neg_items = batch_data[:, 2]
        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))

    def get_statistics(self):
        sta = ""
        sta += 'n_users=%d, n_items=%d\t' % (self.n_users, self.n_items)
        sta += 'n_interactions=%d\t' % (self.n_train + self.n_test)
        sta += 'n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items))
        return sta
