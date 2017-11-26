import tensorflow as tf
import numpy as np
import input_data
import math
import os
import csv
import sys
from pdb import set_trace
from tqdm import tqdm
from util import eprint

def conv_layer(input, num_outputs, kernel_size, stride=1):
    return tf.contrib.layers.conv2d(input, num_outputs=num_outputs,
        kernel_size=kernel_size, stride=stride, weights_regularizer=tf.nn.l2_loss, \
        activation_fn=None)

def pool_layer(input, pool_type):
    if pool_type == 'max':
        return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), \
                padding='SAME')
    elif pool_type == 'avg':
        return tf.reduce_mean(input, axis=[1, 2])

def fc_layer(input, num_outputs):
    return tf.contrib.layers.fully_connected(input, num_outputs, activation_fn=None, \
            weights_regularizer=tf.nn.l2_loss)


class ConvLadderNetwork(object):
    """ This is the Gamma implementation as per the paper """

    def __init__(self):
        self.layers = ('conv', 'maxpool', \
                       'conv', 'conv', 'maxpool', \
                        'conv', 'conv', 'avgpool', \
                        'fc')
        self.L = len(self.layers) - 1
         # For each conv, store number of neurons and kernel size
        # For each fc, store number of neurons
        self.conv_params = {0: (32, [5, 5]), \
                            2: (64, [3, 3]),  \
                            3: (64, [3, 3]),  \
                            5: (128, [3, 3]), \
                            6: (10, [1, 1])}
        self.fc_params = {8: 10}
        self.flattened = [False, False, False, False, False, False, False, True, False]
        self.batch_size = 100
        self.num_labeled = 100
        self.create_compute_graph()

    def bi(self, inits, size, name, scope="ladder_net"):
        with tf.variable_scope(scope):
            return tf.Variable(inits * tf.ones([size]), name=name)

    def create_encoder(self, inputs, noise_std):
        h = inputs + tf.random_normal(tf.shape(inputs)) * noise_std 
        d = {}
        
        def training_batch_norm():
            z = self.join(tf.contrib.layers.batch_norm(z_pre_l, is_training=True), \
                          tf.contrib.layers.batch_norm(z_pre_u, is_training=True))
            z += tf.random_normal(tf.shape(z_pre)) * noise_std
            return z

        def eval_batch_norm():
            z = tf.contrib.layers.batch_norm(z_pre, is_training=False)
            return z
        
        def update_d(d, z, m, v, h, l, use_flattened=False):
            d['labeled']['h'][l], d['unlabeled']['h'][l] = self.split_lu(h)
            if use_flattened:
                d['labeled']['z'][l + 1], d['unlabeled']['z'][l + 1] = self.flat_split_lu(z)
                d['unlabeled']['m'][l + 1], d['unlabeled']['v'][l + 1] = m, v
            else:
                d['labeled']['z'][l + 1], d['unlabeled']['z'][l + 1] = self.split_lu(z)
                d['unlabeled']['m'][l + 1], d['unlabeled']['v'][l + 1] = m, v
            return d

        # The data for labeled and unlabeled examples are stored separately
        d['labeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
        d['unlabeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
        d['labeled']['z'][0], d['unlabeled']['z'][0] = self.split_lu(h)

        for (l, layer_type) in enumerate(self.layers):
            eprint("Layer {}: {}".format(l, layer_type))
            if layer_type == 'conv':
                z_pre = conv_layer(h, self.conv_params[l][0], self.conv_params[l][1])
            elif layer_type == 'maxpool':
                z_pre = pool_layer(h, 'max')
            elif layer_type == 'avgpool':
                z_pre = pool_layer(h, 'avg')
            elif layer_type == 'fc':
                z_pre = fc_layer(h, self.fc_params[l])
            if self.flattened[l]:
                z_pre_l, z_pre_u = self.flat_split_lu(z_pre)  # split labeled and unlabeled examples
            else:
                z_pre_l, z_pre_u = self.split_lu(z_pre)  # split labeled and unlabeled examples
            m, v = tf.nn.moments(z_pre_u, axes=[0])
            z = tf.cond(self.training, training_batch_norm, eval_batch_norm)
            d = update_d(d, z, m, v, h, l, use_flattened=self.flattened[l])
            if layer_type == 'conv':
                eprint(z.get_shape())
                h = tf.nn.relu(z + self.weights["beta"][l])
            elif layer_type == 'fc':
                h = tf.nn.softmax(self.weights['gamma'][l] * (z + self.weights['beta'][l]))
        d['labeled']['h'][self.L + 1], d['unlabeled']['h'][self.L + 1] = self.split_lu(h)
        return h, d

    def create_compute_graph(self):
        self.inputs = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
        self.outputs = tf.placeholder(tf.float32)
        self.training = tf.placeholder(tf.bool)

        self.weights = {'beta': {k : self.bi(0.0, v[0], "beta", scope="transfer_weights") \
                                 for (k, v) in self.conv_params.items()},
                        'gamma': {8: self.bi(1.0, 10, "gamma")}}
        # Hack: Hardcode last beta weight for fully connected + softmax at end.
        self.weights['beta'][8] = self.bi(0.0, 10, "beta", scope="transfer_weights")

        self.join = lambda l, u: tf.concat([l, u], 0)
        self.labeled = lambda x: tf.slice(x, [0, 0, 0, 0], [self.batch_size, -1, -1, -1]) if x is not None else x
        self.unlabeled = lambda x: tf.slice(x, [self.batch_size, 0, 0, 0], [-1, -1, -1, -1]) if x is not None else x
        self.split_lu = lambda x: (self.labeled(x), self.unlabeled(x))
        self.flat_labeled = lambda x: tf.slice(x, [0, 0], [self.batch_size, -1]) if x is not None else x
        self.flat_unlabeled = lambda x: tf.slice(x, [self.batch_size, 0], [-1, -1]) if x is not None else x
        self.flat_split_lu = lambda x: (self.flat_labeled(x), self.flat_unlabeled(x))

        self.noise_std = 0.3
        self.y_c, self.corr = self.create_encoder(self.inputs, self.noise_std)
        self.y, self.clean = self.create_encoder(self.inputs, 0.0)

def main():
    ladder = ConvLadderNetwork()

if __name__=='__main__':
    main()
