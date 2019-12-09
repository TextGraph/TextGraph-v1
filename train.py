import numpy as np
import tensorflow as tf
from utils import *
from model import *
import time
from preData import *
from sklearn.metrics import f1_score

flags = tf.flags
# Model Hyperparameters
flags.DEFINE_string("dataset", "mr", "Dataset (default: mr)")  # "mr", "R8", "R52", "yelp"
flags.DEFINE_string("model", "attention_v2",
                    "The type of neural network(default: attention_v2)")  # attention_v2, attention_v1, GCN(Kipf), attention_v0
flags.DEFINE_float("lr", 0.001, "Learning rate (default: 0.001)")
flags.DEFINE_float("weight", 1e-3, "L2 regularization (default: 1e-3)")
flags.DEFINE_integer("hidden_dim", 128, "Dimensionality of hidden layer (default: 128)")
flags.DEFINE_integer("inputs_dim", 300, "Dimensionality of word embedding (default: 300)")
flags.DEFINE_integer("every_print", 50, "Frequency of printing (default: 50)")
flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
flags.DEFINE_integer("epoch", 200, "Number of training epochs (default: 200)")
flags.DEFINE_integer("words_limits", 150, "Number of words (default: 150)")
flags.DEFINE_string("win_size", "4", "size of slid window (default: 4), it also can be a list to get different graph")  # 4 or 4,5

FLAGS = flags.FLAGS

if FLAGS.model not in ["attention_v2", "attention_v1", "GCN(Kipf)", "attention_v0"]:
    raise Exception("error model")


def select_data(data_name):
    datasets = ["mr", "R8", "R52", "yelp"]
    paraData = [[2, 8, 52, 5], [7108, 5485, 6532, 14000]]
    if data_name in datasets:
        index = datasets.index(data_name)
        return [paraData[0][index], paraData[1][index]]
    else:
        raise Exception("error dataset name")


outputs_dim, split_num = select_data(FLAGS.dataset)
tf.random.set_random_seed(3)
hyperpara = {
    "lr": FLAGS.lr,
    "hidden_dim": FLAGS.hidden_dim,
    "outputs_dim": outputs_dim,
    "every_print": FLAGS.every_print,
    "epoch": FLAGS.epoch,
    "inputs_dim": FLAGS.inputs_dim,
    "weight": FLAGS.weight,
    "batch_size": FLAGS.batch_size,
    "words_limits": FLAGS.words_limits,
    "win_size": list(map(int, FLAGS.win_size.split(","))),
    "seq_length": FLAGS.words_limits,
}
main(hyperpara["win_size"], hyperpara["words_limits"], hyperpara["seq_length"], FLAGS.dataset)
np.set_printoptions(edgeitems=20)
train_mask, train_label, test_mask, test_label, train_index, test_index, train_seq, test_seq, \
dev_masks, dev_index, dev_label, dev_seq = load_data(hyperpara["win_size"], split_num)

print(train_seq.shape)

print(test_mask[0][0].shape)
print(sp.coo_matrix(test_mask[0][0]))
support_num = len(train_mask)
placeholders = {
    "seq_adj": [tf.placeholder(tf.float32, shape=(None, hyperpara["words_limits"], hyperpara["words_limits"])) for _ in
                range(support_num)],
    "seq_inputs": tf.placeholder(tf.int32, shape=(None, hyperpara["words_limits"])),
    "seq_cnn": tf.placeholder(tf.int32, shape=(None, hyperpara["seq_length"])),
    "labels": tf.placeholder(tf.int32, shape=[None, ]),
    "dropout": tf.placeholder_with_default(0.5, shape=()),
    "sp_adj_global": tf.sparse_placeholder(tf.float32),
    "user_node_index": tf.placeholder(tf.int32, shape=[None, ]),
    "global_step": tf.placeholder(tf.int32),
    "lr": tf.placeholder(tf.float32),
    "hyper_node_index": tf.placeholder(tf.int32, shape=(None, hyperpara["words_limits"] - 1))
}
word_emb = get_loc_inputs()

model = GCNNmodel(word_emb, FLAGS.model, placeholders, hyperpara, model_type=FLAGS.model)
sess = tf.Session()
max_acc = 0.


def evaluate(seq_adj, seq_inputs, labels, placeholders, test_cnn, matrix):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(seq_adj, seq_inputs, labels, placeholders, test_cnn)
    feed_dict_val.update({placeholders['dropout']: 0.})
    feed_dict_val.update({placeholders["user_node_index"]: matrix.reshape((-1,))})
    outs_val = sess.run([model.loss, model.accuracy, model.pre_labels, model.user_outputs, model.accuracy_2],
                        feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test), outs_val[2], outs_val[3], outs_val[4]


def batchText(text_batch, seq_adj, seq_inputs, labels, placeholders, test_cnn, matrix):
    m, k = 0, text_batch
    outputs = []
    while k < text_batch:
        out = evaluate(seq_adj[m:k], seq_inputs[m:k], labels[m:k], placeholders, test_cnn[m:k], matrix[m:k])
        outputs.append(out[1])
        m += text_batch
        k += text_batch
    out = evaluate(seq_adj[m:], seq_inputs[m:], labels[m:], placeholders, test_cnn[m:], matrix[m:])
    outputs.append(out[1])
    return np.mean(outputs)


test_mar = genMatrxi(len(test_seq), hyperpara["words_limits"])

delte = find2mark(test_seq)
test_seq = test_seq + test_mar * delte

print("test_seq:", test_seq[4])
sess.run(tf.global_variables_initializer())

cost_val = []
epochs = hyperpara["epoch"]
batch_size = hyperpara["batch_size"]
m, k = 0, batch_size
max_f1 = 0
max_acc_2 = 0
num_user = len(train_mask[0])
e_flag = True
for epoch in range(hyperpara["epoch"]):
    e_flag = True
    while (e_flag):
        if k >= num_user:
            train_adj = []
            for adj in train_mask:
                train_adj.append(adj[m:])
            label_items = train_label[m:]
            train_inputs = train_index[m:]
            train_cnn = train_seq[m:]
            matrix = genMatrxi(len(train_cnn), hyperpara["words_limits"])
            delte = find2mark(train_cnn)
            train_cnn = train_cnn.copy() + matrix * delte
            m = 0
            k = batch_size
            e_flag = False
        else:
            train_adj = []
            for adj in train_mask:
                train_adj.append(adj[m:k])
            label_items = train_label[m:k]
            train_inputs = train_index[m:k]
            train_cnn = train_seq[m:k]
            matrix = genMatrxi(batch_size, hyperpara["words_limits"])
            delte = find2mark(train_cnn)
            train_cnn = train_cnn.copy() + matrix * delte
            k += batch_size
            m += batch_size
        t = time.time()
        feed_dict_train = construct_feed_dict(train_adj, train_inputs, label_items, placeholders, train_cnn)
        feed_dict_train.update({placeholders["dropout"]: 0.5})
        feed_dict_train.update({placeholders["global_step"]: epoch})
        feed_dict_train.update({placeholders["user_node_index"]: matrix.reshape((-1,))})
        outs = sess.run([model.opt_op, model.loss, model.accuracy, model.pre_labels, model.att],
                        feed_dict=feed_dict_train)

    cost, acc, duration, lab, emb, acc2 = evaluate(test_mask, test_index, test_label, placeholders, test_seq, test_mar)
    f1_score_test = f1_score(test_label, lab, average='macro')
    max_f1 = np.maximum(f1_score_test, max_f1)
    max_acc_2 = np.maximum(acc2, max_acc_2)
    if acc > max_acc:
        emb, pre_label = emb, lab
        max_acc = acc
    print("epoch:%04d " % (epoch + 1), "train_loss:{:.5f} ".format(outs[1]), "train_acc:{:.5f}".format(outs[2]),
          "test_loss:{:.5f} ".format(cost), "test_acc:{:.5f}".format(acc), "max_acc:{:.5f}".format(max_acc),
          "f1", "{:.5f}".format(max_f1), "acc2:{:.5}".format(acc2), "max_acc2:{:.5}".format(max_acc_2))
