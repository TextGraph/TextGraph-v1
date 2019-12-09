import tensorflow as tf
import numpy as np
from layer import *


class GCNNmodel:
    def __init__(self, inputs, name, placeholders, hyperpara, model_type="attention_v2"):
        self.layers = []
        self.name = name
        self.activations = []
        self.seq_inputs = placeholders["seq_inputs"]
        self.seq_adj = placeholders["seq_adj"]
        self.dropout = placeholders["dropout"]
        self.labels = placeholders["labels"]
        self.cnn_seq = placeholders["seq_cnn"]
        self.user_node_index = placeholders["user_node_index"]
        self.lr = placeholders["lr"]
        self.loc_outputs = None
        self.accuracy_2 = 0
        self.user_outputs = None
        self.model_type = model_type
        self.optimizer = tf.train.AdamOptimizer(hyperpara["lr"], beta1=0.9)
        self.loss = 0
        self.pre_labels = None
        self.vars = None
        self.uni_emb = 0
        self.accuracy = 0
        self.opt_op = None
        self.hyper_emb = None
        self.cnn_emb = None
        self.w = tf.Variable(initial_value=inputs, name="embeddings", trainable=False)
        self.weight = hyperpara["weight"]
        self.word_emb_gcn = None
        self.output_dim = hyperpara["outputs_dim"]
        self.hidden_dim = hyperpara["hidden_dim"]
        self.input_dims = hyperpara["inputs_dim"]
        self.init_emb = None
        self.att = None
        self.build()

    def _build(self):
        self.layers.append(CnnLayerLast(self.hidden_dim, self.input_dims, self.seq_adj,
                                        dropout=self.dropout, act=tf.nn.relu, name="layer_1", weight_=self.weight,
                                        cnn_seq=self.cnn_seq, model_type=self.model_type))
        self.layers.append(CnnLayerLast(self.hidden_dim, self.hidden_dim, self.seq_adj,
                                        dropout=self.dropout, act=lambda x: x, name="layer_2", weight_=self.weight,
                                        layer_dim=2, cnn_seq=self.cnn_seq, model_type=self.model_type))
        self.layers.append(MeanPoolingLayer(self.output_dim, self.hidden_dim,
                                            dropout=self.dropout, act=lambda x: x, name="layer_4", weight_=self.weight,
                                            user_node_index=self.user_node_index))

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        inputs = tf.nn.embedding_lookup(self.w, self.seq_inputs)
        self.init_emb = inputs
        word_inputs = []
        for i in range(len(self.seq_adj)):
            word_inputs.append(inputs)
        self.activations.append(word_inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.att = self.activations[-2][-1]
        self.user_outputs, self.hyper_emb, self.cnn_emb = self.activations[-1]
        self.word_emb_gcn = self.activations[-2][0][0]
        train_part = self.user_outputs
        self.loss += self._loss(train_part)
        self.accuracy = self._accuracy(train_part)
        self.accuracy_2 = self._ensemble_acc()
        self.opt_op = self.optimizer.minimize(self.loss)

    def _loss(self, logits):
        one_hot_labels = tf.one_hot(self.labels, self.output_dim)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_labels))
        return loss

    def _accuracy(self, logits):
        pre_labels = tf.nn.softmax(logits)
        self.pre_labels = tf.argmax(pre_labels, 1)
        self.labels = tf.cast(self.labels, tf.int32)
        self.pre_labels = tf.cast(self.pre_labels, tf.int32)
        acc = tf.cast(tf.equal(self.pre_labels, self.labels), tf.float32)
        acc_ = tf.reduce_mean(acc)
        return acc_

    def _ensemble_acc(self):
        pre_labels_hyper = tf.nn.softmax(self.hyper_emb)
        pre_labels_cnn = tf.nn.softmax(self.cnn_emb)
        max_labels = tf.maximum(pre_labels_hyper, pre_labels_cnn)
        pre_labels = tf.argmax(max_labels, 1)
        self.labels = tf.cast(self.labels, tf.int32)
        pre_labels = tf.cast(pre_labels, tf.int32)
        acc = tf.cast(tf.equal(pre_labels, self.labels), tf.float32)
        acc_ = tf.reduce_mean(acc)
        return acc_
