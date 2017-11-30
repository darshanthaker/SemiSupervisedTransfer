import tensorflow as tf
from pdb import set_trace
from util import *
import numpy as np
import scipy.io

def conv_layer(input, num_outputs, kernel_size, stride=1):
    return tf.contrib.layers.conv2d(input, num_outputs=num_outputs,
        kernel_size=kernel_size, stride=stride, weights_regularizer=tf.nn.l2_loss)

def pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1),
            padding='SAME')

def fc_layer(input, n, activation_fn=tf.nn.relu):
    return tf.contrib.layers.fully_connected(input, n, activation_fn=activation_fn, \
        weights_regularizer=tf.nn.l2_loss)  

def lrn_layer(input):
    depth_radius = 5
    alpha = 1e-04
    beta = 0.75
    bias = 2.0
    return tf.nn.local_response_normalization(input, depth_radius=depth_radius, \
        alpha=alpha, beta=beta, bias=bias)

class AlexNet(object):

    def __init__(self):
        pass

    def feed_forward(self, input_image, scope='AlexNet'):
        net = {}

        with tf.variable_scope(scope):
            net['conv_1'] = conv_layer(input_image, 64, 11, stride=4)
            net['conv_1'] = lrn_layer(net['conv_1'])
            net['pool_1'] = pool_layer(net['conv_1'])
            net['conv_2'] = conv_layer(net['pool_1'], 192, 5)
            net['conv_2'] = lrn_layer(net['conv_2'])
            net['pool_2'] = pool_layer(net['conv_2'])
            net['conv_3'] = conv_layer(net['pool_2'], 384, 3)
            net['conv_4'] = conv_layer(net['conv_3'], 384, 3)
            net['conv_5'] = conv_layer(net['conv_4'], 250, 3)
            net['pool_3'] = pool_layer(net['conv_5'])
        return net
