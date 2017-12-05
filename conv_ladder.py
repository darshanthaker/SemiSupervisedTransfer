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

def conv_layer(input, num_outputs, kernel_size, stride=1, scope="conv", reuse=None):
    eprint("Reusing with scope {}: {}".format(scope, reuse))
    return tf.contrib.layers.conv2d(input, num_outputs=num_outputs,
        kernel_size=kernel_size, stride=stride, weights_regularizer=tf.nn.l2_loss, \
        activation_fn=None, scope=scope, reuse=reuse)

def pool_layer(input, pool_type):
    if pool_type == 'max':
        return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), \
                padding='SAME')
    elif pool_type == 'avg':
        return tf.reduce_mean(input, axis=[1, 2])

def fc_layer(input, num_outputs, scope="fc", reuse=None):
    return tf.contrib.layers.fully_connected(input, num_outputs, activation_fn=None, \
            weights_regularizer=tf.nn.l2_loss, scope=scope, reuse=reuse)


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
        self.flattened = [False, False, False, False, False, False, False, False, True]
        self.h_flattened = [False, False, False, False, False, False, False, False, True]
        self.batch_size = 100
        self.num_examples = 60000
        self.num_epochs = 150
        self.num_labeled = 100
        self.starter_learning_rate = 0.00002
        self.decay_after = 15  # epoch after which to begin learning rate decay
        self.num_iter = int((self.num_examples//self.batch_size) * self.num_epochs)  # number of loop iterations

        self.create_compute_graph()

    
    def wi(self, shape, name, scope="ladder_net"):
        with tf.variable_scope(scope):
            return tf.Variable(tf.random_normal(shape, name=name), name=name)

    def true_divide(self, var, shape, scope="ladder_net"):
        with tf.variable_scope(scope):
            return var / math.sqrt(shape[0])

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
        
        def update_d(d, z, m, v, h, l, use_flattened=False, use_h_flattened=False):
            if use_h_flattened:
                d['labeled']['h'][l], d['unlabeled']['h'][l] = self.flat_split_lu(h)
            else:
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

        # Corrupt encoder built before clean encoder always.
        reuse = noise_std == 0.0
        if not reuse:
            reuse = None

        debug_z = list()

        for (l, layer_type) in enumerate(self.layers):
            eprint("Layer {}: {}".format(l, layer_type))
            if layer_type == 'conv':
                z_pre = conv_layer(h, self.conv_params[l][0], self.conv_params[l][1], \
                    scope="conv{}".format(l), reuse=reuse)
            elif layer_type == 'maxpool':
                z_pre = pool_layer(h, 'max')
            #elif layer_type == 'avgpool':
            #    z_pre = pool_layer(h, 'avg')
            elif layer_type == 'fc':
                h = tf.contrib.layers.flatten(h)
                z_pre = fc_layer(h, self.fc_params[l], \
                    scope="fc{}".format(l), reuse=reuse)
            if self.flattened[l]:
                z_pre_l, z_pre_u = self.flat_split_lu(z_pre)  # split labeled and unlabeled examples
            else:
                z_pre_l, z_pre_u = self.split_lu(z_pre)  # split labeled and unlabeled examples
            m, v = tf.nn.moments(z_pre_u, axes=[0])
            z = tf.cond(self.training, training_batch_norm, eval_batch_norm)
            d = update_d(d, z, m, v, h, l, use_flattened=self.flattened[l], \
                use_h_flattened=self.h_flattened[l])
            eprint(z.get_shape())
            debug_z.append(z)
            if layer_type == 'conv':
                h = tf.nn.relu(z + self.weights["beta"][l])
            elif layer_type == 'fc':
                h = tf.nn.softmax(self.weights['gamma'][l] * (z + self.weights['beta'][l]))
            else:
                h = z
        d['labeled']['h'][self.L + 1], d['unlabeled']['h'][self.L + 1] = self.flat_split_lu(h)
        return h, d, debug_z

    "gaussian denoising function proposed in the original paper"
    def g_gauss(self, z_c, u, size):
        wi = lambda inits, name: tf.Variable(inits * tf.ones([size]), name=name)
        a1 = wi(0., 'a1')
        a2 = wi(1., 'a2')
        a3 = wi(0., 'a3')
        a4 = wi(0., 'a4')
        a5 = wi(0., 'a5')

        a6 = wi(0., 'a6')
        a7 = wi(1., 'a7')
        a8 = wi(0., 'a8')
        a9 = wi(0., 'a9')
        a10 = wi(0., 'a10')

        mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
        v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10

        z_est = (z_c - mu) * v + mu
        return z_est

    # Decoder.
    def create_decoder(self):
        self.d_cost = []
        z, z_c = self.clean['unlabeled']['z'][self.L + 1], self.corr['unlabeled']['z'][self.L + 1]
        m, v = self.clean['unlabeled']['m'].get(self.L + 1, 0), self.clean['unlabeled']['v'].get(self.L + 1, 1-1e-10)
        u = self.flat_unlabeled(self.y_c)
        u = tf.contrib.layers.batch_norm(u, is_training=False) 
        z_est = self.g_gauss(z_c, u, 10)
        z_est_bn = (z_est - m) / v
        self.d_cost.append((tf.reduce_mean(tf.reduce_sum(\
            tf.square(z_est_bn - z), 1)) / 10) * 0.01)

    def create_compute_graph(self):
        self.inputs = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
        self.outputs = tf.placeholder(tf.float32)
        self.training = tf.placeholder(tf.bool)


        self.weights = {'W_raw': [self.wi((10, 10), "W")], \
                        'beta': {k : self.bi(0.0, v[0], "beta", scope="transfer_weights") \
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
        eprint("=== Corrupted Encoder ===")
        self.y_c, self.corr, _ = self.create_encoder(self.inputs, self.noise_std)

        eprint("=== Clean Encoder ===")
        self.y, self.clean, _ = self.create_encoder(self.inputs, 0.0)

        eprint("=== Decoder ===")
        self.create_decoder()

        self.u_cost = tf.add_n(self.d_cost)

        self.y_N = self.flat_labeled(self.y_c)
        self.cost = -tf.reduce_mean(tf.reduce_sum(self.outputs*tf.log(self.y_N), 1))  # supervised cost
        self.loss = self.cost + self.u_cost  # total cost
        self.loss = self.cost

        self.pred_cost = -tf.reduce_mean(tf.reduce_sum(self.outputs*tf.log(self.y), 1))  # cost used for prediction

        self.train_correct_prediction = tf.equal(tf.argmax(self.y_N, 1), tf.argmax(self.outputs, 1))  # no of correct predictions
        self.train_accuracy = tf.reduce_mean(tf.cast(self.train_correct_prediction, "float")) * tf.constant(100.0)

        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.outputs, 1))  # no of correct predictions
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float")) * tf.constant(100.0)

        self.learning_rate = tf.Variable(self.starter_learning_rate, trainable=False)
        bounds = [30, 200, 300]
        values = [1e-2, 1e-3, 1e-4, 1e-5]
        self.step_op = tf.Variable(0, name='step', trainable=False)
        #self.step_op = None
        self.lr = tf.train.piecewise_constant(self.step_op, bounds, values)
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.step_op)
        #self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.step_op)

    def train(self):
        eprint("===  Loading Data ===")
        mnist = input_data.read_data_sets("MNIST_data", n_labeled=self.num_labeled, one_hot=True, flatten=False)

        var_dict = dict()
        for k in self.conv_params.keys():
            var_dict[self.weights['beta'][k].name[:-2]] = self.weights['beta'][k]
        var_dict[self.weights['beta'][8].name[:-2]] = self.weights['beta'][8]

        conv_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "conv")
        fc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fc")
        for c in conv_vars:
            var_dict[c.name[:-2]] = c
        for f in fc_vars:
            var_dict[f.name[:-2]] = f
         
        saver = tf.train.Saver(var_dict)

        eprint("===  Starting Session ===")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        i_iter = 0

        ckpt = tf.train.get_checkpoint_state('conv_checkpoints/')  # get latest checkpoint (if any)
        if ckpt and ckpt.model_checkpoint_path:
            # if checkpoint exists, restore the parameters and set epoch_n and i_iter
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            epoch_n = int(ckpt.model_checkpoint_path.split('-')[1])
            i_iter = (epoch_n+1) * (self.num_examples//self.batch_size)
            eprint("Restored Epoch ", epoch_n)
        else:
            # no checkpoint exists. create checkpoints directory if it does not exist.
            if not os.path.exists('conv_checkpoints'):
                os.makedirs('conv_checkpoints')
            init = tf.global_variables_initializer()
            self.sess.run(init)

        eprint("=== Training ===")
        eprint("Initial Accuracy: ", self.sess.run(self.accuracy, feed_dict={self.inputs: mnist.test.images, self.outputs: mnist.test.labels, self.training: False}), "%")
        losses = list()
        for i in range(i_iter, self.num_iter):
            images, labels = mnist.train.next_batch(self.batch_size)
            acc, loss, _ = self.sess.run([self.train_accuracy, self.loss, self.train_step], feed_dict={self.inputs: images, self.outputs: labels, self.training: True})
            losses.append(loss)
            #eprint("[{}] Learning Rate: {}, Train Accuracy: {}, Unsupervised Loss: {}, Supervised Loss: {}".format(i, lr, acc, u_loss, s_loss))
            if i % 10 == 0:
                eprint("[{}] Loss = {}, Test Accuracy: ".format(i, np.mean(losses)), self.sess.run(self.accuracy, feed_dict={self.inputs: mnist.test.images, self.outputs: mnist.test.labels, self.training: False}), "%")
                losses = list()
            if (i > 1) and ((i+1) % (self.num_iter//self.num_epochs) == 0):
                epoch_n = i//(self.num_examples//self.batch_size)
                if (epoch_n+1) >= self.decay_after:
                    # decay learning rate
                    learning_rate = starter_learning_rate * ((num_epochs - epoch_n) / (num_epochs - decay_after))
                    ratio = 1.0 * (self.num_epochs - (epoch_n+1))  # epoch_n + 1 because learning rate is set for next epoch
                    ratio = max(0, ratio / (self.num_epochs - self.decay_after))
                    self.sess.run(self.learning_rate.assign(self.starter_learning_rate * ratio))
                saver.save(self.sess, 'conv_checkpoints/model.ckpt', epoch_n)
                # print "Epoch ", epoch_n, ", Accuracy: ", sess.run(accuracy, feed_dict={inputs: mnist.test.images, outputs:mnist.test.labels, training: False}), "%"
                with open('train_log', 'a') as train_log:
                    # write test accuracy to file "train_log"
                    train_log_w = csv.writer(train_log)
                    log_i = [epoch_n] + self.sess.run([self.accuracy], feed_dict={self.inputs: mnist.test.images, self.outputs: mnist.test.labels, self.training: False})
                    train_log_w.writerow(log_i)

        
        eprint("Final Accuracy: ", self.sess.run(self.accuracy, feed_dict={self.inputs: mnist.test.images, self.outputs: mnist.test.labels, self.training: False}), "%")
        self.sess.close()

def main():
    ladder = ConvLadderNetwork()
    ladder.train()

if __name__=='__main__':
    main()
