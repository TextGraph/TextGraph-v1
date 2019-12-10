import tensorflow as tf
import numpy as np

tf.set_random_seed(10086)


def glorot(shape, name=None):
    print(shape)
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    # initial= tf.random.truncated_normal(shape)
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def dot(x, y, sparse=False):
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class MeanPoolingLayer:
    def __init__(self, output_dim, input_dim,
                 dropout=0., act=tf.nn.relu, name="layer_1", weight_=1e-4, user_node_index=None):
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = act
        self.name = name
        self.weight = weight_
        self.input_dim = input_dim
        self.user_node_index = user_node_index

    def _call(self, inputs):
        word_outputs, cnn_embs, _ = inputs
        word_outputs_all = tf.concat(word_outputs, 1)
        pre2emb = tf.reshape(word_outputs_all, (-1, self.input_dim * len(word_outputs)))
        hyper_out = tf.nn.embedding_lookup(pre2emb, self.user_node_index)
        cnn_emb_out = tf.concat(cnn_embs, 1)
        hyper_out_drop = tf.nn.dropout(hyper_out, 1 - self.dropout)
        cnn_emb_out_drop = tf.nn.dropout(cnn_emb_out, 1 - self.dropout)
        dense2 = tf.keras.layers.Dense(self.output_dim, self.act,
                                       kernel_regularizer=tf.keras.regularizers.l2(self.weight), name="words")
        h1_c_2 = dense2(hyper_out_drop)
        h2_c_2 = dense2(cnn_emb_out_drop)
        outputs = h1_c_2 + 0.5 * h2_c_2
        return outputs, h1_c_2, h2_c_2

    def __call__(self, inputs):
        return self._call(inputs)


class CnnLayerLast:
    def __init__(self, output_dim, input_dim, seq_adj,
                 dropout=0., act=tf.nn.relu, name="layer_1", weight_=1e-4, layer_dim=0, cnn_seq=None,
                 model_type="attention_v2"):
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = act
        self.name = name
        self.weight = weight_
        self.seq_adj = seq_adj
        self.input_dim = input_dim
        self.layer_dim = layer_dim
        self.cnn_seq = cnn_seq
        self.model_type = model_type

    def _call(self, inputs):
        support_num = len(self.seq_adj)
        cnn_layers = []
        real_adj = tf.zeros(0)
        dense_words = []
        dense_seq = []
        seq_emb = 0
        ker_size = [3, 4, 5]
        for i in range(support_num):
            dense_words.append(
                tf.keras.layers.Dense(self.output_dim, self.act, name=self.name + "dense_words_" + str(i),
                                      use_bias=False,
                                      kernel_regularizer=tf.keras.regularizers.l2(self.weight),
                                      kernel_initializer="glorot_normal"))
            dense_seq.append(tf.keras.layers.Dense(self.output_dim, self.act, name=self.name + "cnn_" + str(i),
                                                   kernel_regularizer=tf.keras.regularizers.l2(self.weight),
                                                   kernel_initializer="glorot_normal"))
        if self.layer_dim > 0:
            word_embs, seqs, _ = inputs
        else:
            word_embs = inputs
        word_outputs = []
        seq_outputs = []
        for i in range(support_num):

            word_emb = tf.nn.dropout(word_embs[i], 1 - self.dropout)
            dims = tf.shape(word_emb)
            if self.model_type == "attention_v1":
                att_out = self.self_att(word_emb, self.output_dim, self.seq_adj[i])
                words_embs = tf.reshape(att_out, (dims[0] * dims[1], self.input_dim))

            elif self.model_type == "GCN(Kipf)":
                pre = tf.matmul(self.seq_adj[i], word_emb)
                pre_re = tf.reshape(pre, (dims[0] * dims[1], self.input_dim))
                words_embs = dense_words[i](pre_re)

            elif self.model_type == "attention_v2":
                seq_fts = tf.keras.layers.Conv1D(self.output_dim, 1, activation=self.act,
                                                 name=self.name + "cnn_" + str(i),
                                                 kernel_regularizer=tf.keras.regularizers.l2(self.weight),
                                                 kernel_initializer="glorot_normal", use_bias=False)(word_emb)
                f_1 = tf.layers.conv1d(seq_fts, 1, 1, activation=tf.nn.sigmoid,
                                       kernel_regularizer=tf.keras.regularizers.l2(self.weight))
                f_2 = tf.layers.conv1d(seq_fts, 1, 1, activation=tf.nn.sigmoid,
                                       kernel_regularizer=tf.keras.regularizers.l2(self.weight))
                logits = f_1 + tf.transpose(f_2, [0, 2, 1])
                temp_var = tf.multiply(self.seq_adj[i], logits)
                real_adj = temp_var
                words_embs = tf.reshape(tf.matmul(temp_var, seq_fts), (dims[0] * dims[1], self.input_dim))

            else:
                seq_fts = tf.keras.layers.Conv1D(self.output_dim, 1, activation=self.act,
                                                 name=self.name + "cnn_" + str(i),
                                                 kernel_regularizer=tf.keras.regularizers.l2(self.weight),
                                                 kernel_initializer="glorot_normal", use_bias=False)(word_emb)
                f_1 = tf.layers.conv1d(seq_fts, 1, 1, activation=tf.nn.sigmoid)
                temp_var = tf.multiply(self.seq_adj[i], f_1)
                self.at_matrix = temp_var
                words_embs = tf.reshape(tf.matmul(temp_var, seq_fts), (dims[0] * dims[1], self.input_dim))

            outputs = tf.reshape(words_embs, (dims[0], dims[1], self.output_dim))
            if self.layer_dim > 1:
                kera_conv = []
                for k in range(len(ker_size)):
                    kera_conv.append(tf.keras.layers.Conv1D(128, ker_size[k], activation=lambda x: x,
                                                            name=self.name + "cnn_" + str(i) + "_" + str(k),
                                                            kernel_regularizer=tf.keras.regularizers.l2(self.weight),
                                                            kernel_initializer="glorot_normal"))
                cnn_layers.append(kera_conv)
                cnn_input = tf.nn.embedding_lookup(tf.nn.tanh(words_embs), self.cnn_seq)
                pre_ = tf.reshape(cnn_input, (dims[0], tf.shape(self.cnn_seq)[1], self.input_dim))
                seq_emb_list = []
                for k in range(len(ker_size)):
                    seq_emb = cnn_layers[i][k](pre_)
                    seq_emb = tf.keras.layers.GlobalMaxPooling1D()(seq_emb)
                    seq_emb_list.append(seq_emb)
                seq_emb = tf.concat(seq_emb_list, 1)
                seq_emb = tf.reshape(dense_seq[i](seq_emb), (dims[0], self.output_dim))
            word_outputs.append(outputs)
            seq_outputs.append(seq_emb)
        return [word_outputs, seq_outputs, real_adj]

    def __call__(self, inputs):
        return self._call(inputs)

    def self_att(self, inputs, att_dims, adj):
        seq = tf.keras.layers.Conv1D(att_dims, 1, activation=self.act, name="q", use_bias=False)(inputs)
        Q = tf.keras.layers.Conv1D(32, 1, activation=lambda x:x, name="k")(seq)
        K = tf.keras.layers.Conv1D(32, 1, activation=lambda x:x, name="v")(seq)
        att = tf.multiply(tf.nn.sigmoid(tf.matmul(Q, tf.transpose(K, [0, 2, 1]))/np.sqrt(32.)), adj)
        output = tf.matmul(att, seq)
        return output
