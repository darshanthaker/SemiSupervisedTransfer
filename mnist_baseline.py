import tensorflow as tf
import numpy as np
import math
import os
from util import eprint
import util
from pdb import set_trace
import input_data

def fc_layer(input, n, activation_fn=tf.nn.relu):
    return tf.contrib.layers.fully_connected(input, n, activation_fn=activation_fn, \
        weights_regularizer=tf.nn.l2_loss)  

class MNISTBaseline(object):


    def __init__(self):
        self.layer_sizes = [784, 1000, 500, 250, 250, 250, 10]
        self.L = len(self.layer_sizes) - 1
        self.create_compute_graph()

    def bi(self, inits, size, name, scope="mnist"):
        with tf.variable_scope(scope):
            return tf.Variable(inits * tf.ones([size]), name=name)

    def wi(self, shape, name, scope="mnist"):
        with tf.variable_scope(scope):
            return tf.Variable(tf.random_normal(shape, name=name), name=name)

    def true_divide(self, var, shape, scope="mnist"):
        with tf.variable_scope(scope):
            return var / math.sqrt(shape[0])

    def create_compute_graph(self):
        self.inputs = tf.placeholder(tf.float32, shape=(None, self.layer_sizes[0]))
        self.labels = tf.placeholder(tf.int64)
        self.o_labels = tf.one_hot(self.labels, 10)
        self.training = tf.placeholder(tf.bool)

        # Shapes of linear layers.
        shapes = list(zip(self.layer_sizes[:-1], self.layer_sizes[1:]))

        self.weights = {'W_raw': [self.wi(s, "W", scope="transfer_weights") \
                              for s in shapes],  # Encoder weights
                   # batch normalization parameter to shift the normalized value
                   'beta': [self.bi(0.0, self.layer_sizes[l+1], "beta", scope="transfer_weights") \
                            for l in range(self.L)],
                   # batch normalization parameter to scale the normalized value
                   'gamma': [self.bi(1.0, self.layer_sizes[l+1], "beta") for l in range(self.L)]}

        self.weights['W'] = [self.true_divide(w, s, scope="transfer_weights") \
                             for (w, s) in zip(self.weights['W_raw'], shapes)]
        
        net = {}
        net['fc0'] = self.inputs

        for l in range(1, self.L + 1):
            prev = 'fc{}'.format(l - 1)
            curr = 'fc{}'.format(l)
   
            net[curr] = tf.matmul(net[prev], self.weights['W'][l - 1]) 
            if l == self.L:
                net[curr] = tf.nn.softmax(self.weights['gamma'][l - 1] * \
                    (net[curr] + self.weights['beta'][l - 1]))
            else:
                net[curr] = tf.nn.relu(net[curr] + self.weights['beta'][l - 1])

        net['output'] = net['fc{}'.format(self.L)]
        net['predicted'] = tf.cast(tf.argmax( \
                    net['output'], axis=-1), tf.int64)

        eprint( "Total number of variables used ", np.sum([v.get_shape().num_elements() for v in tf.trainable_variables()]))

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=net['output'], labels=self.o_labels))
        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss = self.loss + 1e-6 * reg_loss
        bounds = [5000, 10000]
        values = [1e-3, 1e-4, 1e-5]
        step_op = tf.Variable(0, name='step', trainable=False)
        lr_op = tf.train.piecewise_constant(step_op, bounds, values)
        self.minimizer = tf.train.AdamOptimizer(lr_op).minimize(self.loss)
        self.correct = tf.equal(net['predicted'], self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

    def train(self, batch_size):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
        mnist = input_data.sl_read_data_sets("MNIST_data", one_hot=False)
        
        var_dict = dict()
        for l in range(self.L):
            var_dict[self.weights['W_raw'][l].name[:-2]] = self.weights['W_raw'][l]
            var_dict[self.weights['beta'][l].name[:-2]] = self.weights['beta'][l]
        saver = tf.train.Saver(var_dict)

        ckpt = tf.train.get_checkpoint_state('baseline_checkpoints/')  # get latest checkpoint (if any)
        if ckpt and ckpt.model_checkpoint_path:
            # if checkpoint exists, restore the parameters and set epoch_n and i_iter
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            epoch_n = int(ckpt.model_checkpoint_path.split('-')[1])
            eprint("Restored epoch {}".format(epoch_n))
        else:
            # no checkpoint exists. create checkpoints directory if it does not exist.
            if not os.path.exists('baseline_checkpoints'):
                os.makedirs('baseline_checkpoints')
            init = tf.global_variables_initializer()
            self.sess.run(init)

        eprint("Initial Accuracy: ", self.sess.run(self.accuracy, feed_dict={self.inputs: mnist.test.images, self.labels: mnist.test.labels, self.training: False}), "%")
        for epoch in range(10000):
            accs = list()
            losses = list()
            images, labels = mnist.train.next_batch(batch_size)
            feed_dict = {self.inputs: images, self.labels: labels, self.training: True}
            acc, loss, _ = self.sess.run([self.accuracy, self.loss, self.minimizer], \
                feed_dict=feed_dict) 
            accs.append(acc)
            losses.append(loss)
            #eprint("[{}] Accuracy: {}, Loss = {}".format(epoch, acc, np.mean(los)))

            if epoch % 100 == 0 and epoch != 0:
                saver.save(self.sess, 'baseline_checkpoints/model.ckpt', epoch)
                eprint("[{}] Train Accuracy: {}, Loss: {}".format(epoch, np.mean(accs), np.mean(losses)))
                accs = list()
                losses = list()
                eprint("[{}] Test Accuracy: ".format(epoch), self.sess.run(self.accuracy, feed_dict={self.inputs: mnist.test.images, self.labels: mnist.test.labels, self.training: False}), "%")
        self.sess.close()

def main():
    baseline = MNISTBaseline()
    baseline.train(64)

if __name__=="__main__":
    main()

