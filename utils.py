import numpy as np
import pickle
import scipy.sparse as sp
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator


def load_data(win_size, dataSize):
    np.random.seed(222)
    seq_lists = []
    for i in win_size:
        seq_lists.append(open_file("seq_list_" + str(i)))
    user_index = open_file("user_index")
    labels = open_file("label")
    sequences = open_file("cnn_seq")
    num_user = len(user_index)
    print("user_num:", num_user)
    split_num = dataSize
    permutation = np.random.permutation(int(split_num - 0.1 * split_num))
    train_masks = []
    test_masks = []
    dev_masks = []
    for seq_list in seq_lists:
        nor_seq_list = []
        for matrx in seq_list:
            item_sp = normalize_adj(matrx).toarray()
            item_ = item_sp[np.newaxis, :]
            nor_seq_list.append(item_)
        nor_seq_ = np.concatenate(nor_seq_list, 0)
        train_mask = nor_seq_[:int(split_num - 0.1 * split_num)]
        train_mask = train_mask[permutation, :, :]
        train_masks.append(train_mask.copy())
        test_mask = nor_seq_[split_num:]
        dev_mask = nor_seq_[int(split_num - 0.1 * split_num):split_num]
        dev_masks.append(dev_mask.copy())
        test_masks.append(test_mask.copy())
    labels = np.array(labels).reshape((-1,))
    matrx_user_index = np.array(user_index)
    sequences_index = np.array(sequences)
    train_label = labels[:int(split_num - 0.1 * split_num)]
    train_index = matrx_user_index[:int(split_num - 0.1 * split_num)]
    train_seq = sequences_index[:int(split_num - 0.1 * split_num)]
    train_label = train_label[permutation]
    train_index = train_index[permutation]
    train_seq = train_seq[permutation]
    test_label = labels[split_num:]
    test_index = matrx_user_index[split_num:]
    test_seq = sequences_index[split_num:]
    dev_label = labels[int(split_num - 0.1 * split_num):split_num]
    dev_index = matrx_user_index[int(split_num - 0.1 * split_num):split_num]
    dev_seq = sequences_index[int(split_num - 0.1 * split_num):split_num]
    return train_masks, train_label, test_masks, test_label, train_index, test_index, train_seq, test_seq, dev_masks, dev_index, dev_label, dev_seq


def construct_feed_dict(seq_adj, seq_inputs, labels, placeholders, cnn_seq):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['seq_adj'][i]: seq_adj[i] for i in range(len(seq_adj))})
    feed_dict.update({placeholders['seq_inputs']: seq_inputs})
    feed_dict.update({placeholders['seq_cnn']: cnn_seq})
    feed_dict.update({placeholders['labels']: labels})

    return feed_dict


def preprocess_adj_(adj):
    rowsum = np.array(adj.sum(1)).astype(np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    adj_pre = r_mat_inv.dot(adj)
    return adj_pre


def open_file(name):
    file = open("../data/" + name + ".pkl", "rb")
    return pickle.load(file)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def preprocess_adj(adj):
    adj_dig = adj - np.diag(adj.diagonal())
    adj_dig = sp.coo_matrix(adj_dig)
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj_dig + sp.eye(adj_dig.shape[0]))
    return adj_normalized


def get_loc_inputs(size=0):
    with open("../data/inputs_loc.pkl", "rb") as f3:
        inputs_loc = pickle.load(f3)
    # inputs_loc = np.random.standard_normal((inputs_loc.shape[0],300)).astype(np.float32)
    return inputs_loc


def adj_to_bias(adj):
    adj_all = []
    for i in range(adj.shape[0]):
        sp_adj = sp.coo_matrix(adj[i])
        sp_adj.data = np.ones(sp_adj.data.shape)
        mt = adj.toarray() - 1.0
        adj_mt = 1e9 * mt
        adj_all.append(adj_mt)
    return np.array(adj_all)


def t_sneVis(emb, label, filename):
    plt.figure(figsize=(3.34, 2.5))
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.yaxis.set_major_locator(MultipleLocator(25))
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('times') for label in labels]
    plt.yticks(size=7)
    plt.xticks(size=7)
    fea = TSNE(n_components=2, perplexity=40).fit_transform(emb)
    cls = np.unique(label)
    fea_num = [fea[label == i] for i in cls]
    for i, f in enumerate(fea_num):
        plt.scatter(f[:, 0], f[:, 1], label=cls[i], marker='.')
    plt.tight_layout()
    plt.savefig(filename + '.pdf', bbox_inches='tight', dpi=300, pad_inches=0.05)
    plt.show()


def genMatrxi(length, seq_len):
    list_matrix = [seq_len * i for i in range(length)]
    matrxi = np.array(list_matrix).reshape((-1, 1))
    return matrxi


def find2mark(matrix):
    new_matrix = matrix.copy()
    new_matrix[matrix != -1] = 1
    new_matrix[matrix == -1] = 0
    return new_matrix
