import tensorflow as tf
import numpy as np
import math
from util import eprint
import util
from pdb import set_trace

# Transfer only between two tasks currently.
class ProgressiveNetwork(object):

    def __init__(self, from_arch, to_arch):
        self.from_arch = from_arch
        self.to_arch = to_arch
        if self.from_arch == "ladder":
            self.layer_sizes = [3072, 1000, 500, 250, 250, 250, 10]
            self.L = len(self.layer_sizes) - 1
        self.from_input_loader = util.InputLoader('svhn')
        self.to_num_classes = self.from_input_loader.get_num_classes()
        self.create_compute_graph()

    def wi(self, shape, name, scope="transfer_weights"):
        with tf.variable_scope(scope):
            return tf.Variable(tf.random_normal(shape, name=name), name=name)

    def true_divide(self, var, shape, scope="transfer_weights"):
        with tf.variable_scope(scope):
            return var / math.sqrt(shape[0])

    def bi(self, inits, size, name, scope="transfer_weights"):
        with tf.variable_scope(scope):
            return tf.Variable(inits * tf.ones([size]), name=name)

    def ladder_encoder(self):
        net = {}
        net['fc0'] = self.inputs

        sess = tf.Session()

        shapes = list(zip(self.layer_sizes[:-1], self.layer_sizes[1:]))

        self.weights = {'W_raw': [self.wi(s, "W") \
                                  for (i, s) in enumerate(shapes)], \
                        'beta': [self.bi(0.0, self.layer_sizes[l + 1], \
                                'beta') for l in range(self.L)]}

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('weight_checkpoints')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            epoch_n = int(ckpt.model_checkpoint_path.split('-')[1])
            eprint("Restored Epoch ", epoch_n)
        eprint("Restored weights from file") 

        self.weights['W'] = [self.true_divide(w, s, scope="transfer_weights") \
                             for (w, s) in zip(self.weights['W_raw'], shapes)]
        
        for l in range(1, self.L + 1):
            layer = 'fc{}'.format(l)
            prev_layer = net['fc{}'.format(l - 1)]
            net[layer] = tf.matmul(prev_layer, self.weights['W'][l - 1])
            net[layer] = tf.layers.batch_normalization(net[layer], training=False) 
            net[layer] = tf.nn.relu(net[layer] + self.weights['beta'][l - 1])
        return net

    def fc_adapter(self, inputs, dim):
        # TODO(dbthaker): Add learnable scalar coefficient
        return tf.contrib.layers.fully_connected(inputs, dim, activation_fn=tf.nn.relu)

    def fc_decoder(self):
        net = {}
        net['fc0'] = self.inputs

        net['fc1'] = tf.contrib.layers.fully_connected(net['fc0'], 3072, \
                weights_regularizer=tf.nn.l2_loss)
        net['fc2'] = tf.contrib.layers.fully_connected(net['fc1'], 1000, \
                weights_regularizer=tf.nn.l2_loss)
        net['fc3'] = tf.contrib.layers.fully_connected(net['fc2'], 500, \
                weights_regularizer=tf.nn.l2_loss)
        net['fc4'] = tf.contrib.layers.fully_connected(net['fc3'], 250, \
                weights_regularizer=tf.nn.l2_loss)
        net['fc5'] = tf.contrib.layers.fully_connected(net['fc4'], 250, \
                weights_regularizer=tf.nn.l2_loss)
        net['fc6'] = tf.contrib.layers.fully_connected(net['fc5'], 250, \
                weights_regularizer=tf.nn.l2_loss)
        net['output'] = tf.contrib.layers.fully_connected(net['fc6'], self.to_num_classes, \
                weights_regularizer=tf.nn.l2_loss)

        net['predicted'] = tf.cast(tf.argmax(tf.nn.softmax(net['output']), axis=-1), tf.int64)

        return net

    def create_compute_graph(self):
        self.inputs = tf.placeholder(tf.float32, shape=(None, 3072))
        self.raw_labels = tf.placeholder(tf.int64, shape=(None))
        self.labels = tf.one_hot(self.raw_labels, self.to_num_classes)
        #ladder_encoder = self.ladder_encoder()
        fc_decoder = self.fc_decoder()

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=fc_decoder['output'], labels=self.labels))

        self.minimizer = tf.train.AdamOptimizer(1e-2).minimize(self.loss)
        self.correct = tf.equal(fc_decoder['predicted'], self.raw_labels)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

    def train(self, batch_size):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
        input_loader = self.from_input_loader
       
        for epoch in range(100):
            train_X, train_y = input_loader.get_train_data()
            np.random.seed(epoch)
            indices = np.arange(train_y.shape[0])
            np.random.shuffle(indices)
            train_X = train_X[:, :, :, indices]
            train_y = train_y[indices]
            accs = list()
            losses = list()
            for i in range(0, train_X.shape[3]-batch_size + 1, batch_size):
                batch_images, batch_labels_raw = \
                    train_X[:, :, :, i:i+batch_size], \
                    train_y[i:i+batch_size]
                batch_images = batch_images.reshape((32 * 32 * 3, batch_size)).T
                feed_dict = {self.inputs: batch_images, self.raw_labels: batch_labels_raw}
                acc, loss, _ = self.sess.run([self.accuracy, self.loss, self.minimizer], \
                    feed_dict=feed_dict) 
                accs.append(acc)
                losses.append(loss)
            eprint("[{}] Accuracy: {}, Loss = {}".format(epoch, np.mean(accs), np.mean(losses)))

            if epoch % 10 == 0:
                test_X, test_y = input_loader.get_test_data()
                test_X = test_X.reshape((32 * 32 * 3, -1)).T
                feed_dict = {self.inputs: test_X, self.raw_labels: test_y}
                acc = self.sess.run([self.accuracy], feed_dict=feed_dict)
                eprint("[%d] Test accuracy: {}".format(acc)) 

def main():
    prog = ProgressiveNetwork("ladder", "tmp")
    prog.train(64)

if __name__=="__main__":
    main()
