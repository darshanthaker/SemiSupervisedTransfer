import tensorflow as tf
import numpy as np
import math
from util import eprint
from alexnet import AlexNet
import util
from pdb import set_trace

def fc_layer(input, n, activation_fn=tf.nn.relu):
    return tf.contrib.layers.fully_connected(input, n, activation_fn=activation_fn, \
        weights_regularizer=tf.nn.l2_loss)  

# Transfer only between two tasks currently.
class ProgressiveNetwork(object):

    def __init__(self, from_arch, to_arch):
        self.from_arch = from_arch
        self.to_arch = to_arch
        if self.from_arch == "ladder":
            self.layer_sizes = [784, 1000, 500, 250, 250, 250, 10]
            self.L = len(self.layer_sizes) - 1
        self.from_input_loader = util.InputLoader('svhn')
        self.to_num_classes = self.from_input_loader.get_num_classes()
        self.create_compute_graph()

    def wi(self, shape, name, scope="transfer_weights", trainable=False):
        with tf.variable_scope(scope):
            return tf.Variable(tf.random_normal(shape, name=name), name=name, \
                trainable=trainable)

    def true_divide(self, var, shape, scope="transfer_weights"):
        with tf.variable_scope(scope):
            return var / math.sqrt(shape[0])

    def bi(self, inits, size, name, scope="transfer_weights", trainable=False):
        with tf.variable_scope(scope):
            return tf.Variable(inits * tf.ones([size]), name=name, \
                trainable=trainable)

    def load_ladder_weights(self):
        sess = tf.Session()
        shapes = list(zip(self.layer_sizes[:-1], self.layer_sizes[1:]))
        self.weights = {'W_raw': [self.wi(s, "W", trainable=False) \
                                  for (i, s) in enumerate(shapes)], \
                        'beta': [self.bi(0.0, self.layer_sizes[l + 1], \
                                'beta', trainable=False) for l in range(self.L)]}

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('weight_checkpoints')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            epoch_n = int(ckpt.model_checkpoint_path.split('-')[1])
            eprint("Restored Epoch ", epoch_n)
        eprint("Restored weights from file") 

        self.weights['W'] = [self.true_divide(w, s, scope="transfer_weights") \
                             for (w, s) in zip(self.weights['W_raw'], shapes)]

    def fc_adapter(self, inputs, dim):
        # TODO(dbthaker): Add learnable scalar coefficient
        return tf.contrib.layers.fully_connected(inputs, dim, activation_fn=tf.nn.relu)

    def fc_decoder(self):
        net = {}
        self.gs_inputs = tf.image.rgb_to_grayscale(self.inputs)
        self.res_inputs = tf.image.resize_images(self.gs_inputs, (28, 28))
        net['fc0'] = tf.contrib.layers.flatten(self.res_inputs)

        net['fc1'] = tf.contrib.layers.fully_connected(net['fc0'], 784, \
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
        self.inputs = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
        self.raw_labels = tf.placeholder(tf.int64, shape=(None))
        self.labels = tf.one_hot(self.raw_labels, self.to_num_classes)

        if self.from_arch == 'ladder' and self.to_arch == 'alex':
            self.load_ladder_weights()
            alexnet_class = AlexNet()
            alexnet = alexnet_class.feed_forward(self.inputs)

            with tf.variable_scope('progressive_net'):
                alexnet['pool_3'] = tf.contrib.layers.flatten(alexnet['pool_3'])

                alexnet['fc_1'] = fc_layer(alexnet['pool_3'], 250)
                alexnet['fc_1'] = tf.layers.batch_normalization(alexnet['fc_1'])
                alexnet['fc_1'] = tf.layers.dropout(alexnet['fc_1'], 0.5) 

                ladder1 = tf.matmul(alexnet['pool_3'], self.weights['W'][self.L - 3])
                ladder1 = tf.layers.batch_normalization(ladder1, training=False) 
                ladder1 = tf.nn.relu(ladder1 + self.weights['beta'][self.L - 3])

                first_out = fc_layer(alexnet['fc_1'], 250)
                second_out = fc_layer(ladder1, 250)
                alexnet['fc_2'] = first_out + second_out
                alexnet['fc_2'] = tf.layers.batch_normalization(alexnet['fc_2'])
                alexnet['fc_2'] = tf.layers.dropout(alexnet['fc_2'], 0.5) 
                ladder2 = tf.matmul(ladder1, self.weights['W'][self.L - 2])
                ladder2 = tf.layers.batch_normalization(ladder2, training=False) 
                ladder2 = tf.nn.relu(ladder2 + self.weights['beta'][self.L - 2])

                first_out = fc_layer(alexnet['fc_2'], self.to_num_classes, activation_fn=None)
                second_out = fc_layer(ladder2, self.to_num_classes, activation_fn=None)
                alexnet['output'] = first_out + second_out
                alexnet['predicted'] = tf.cast(tf.argmax(\
                    tf.nn.softmax(alexnet['output']), axis=-1), tf.int64)
            net = alexnet
        elif self.from_arch == 'ladder' and self.to_arch == 'fc':
            self.gs_inputs = tf.image.rgb_to_grayscale(self.inputs)
            self.res_inputs = tf.image.resize_images(self.gs_inputs, (28, 28))
            self.load_ladder_weights()
            from_net = {}
            to_net = {}
            to_net['fc0'] = tf.contrib.layers.flatten(self.res_inputs)
            from_net['fc0'] = tf.contrib.layers.flatten(self.res_inputs)

            for l in range(1, self.L + 2):
                prev = 'fc{}'.format(l - 1)
                curr = 'fc{}'.format(l)

                if l != self.L + 1:
                    from_net[curr] = tf.matmul(from_net[prev], self.weights['W'][l - 1])
                    from_net[curr] = tf.layers.batch_normalization(from_net[curr], training=False)
                    from_net[curr] = tf.nn.relu(from_net[curr] + self.weights['beta'][l - 1])
                first_out = fc_layer(to_net[prev], self.layer_sizes[l - 1])
                #print("Ladder {} -> Ladder {}".format(prev, curr))
                if l != 1:
                    second_out = fc_layer(from_net[prev], self.layer_sizes[l - 1])
                    to_net[curr] = first_out + second_out
                    #print("Ladder {} -> New {} {} + New {} -> New {} {}".format(\
                    #    prev, curr, self.layer_sizes[l - 1], \
                    #    prev, curr, self.layer_sizes[l - 1]))
                else:
                    to_net[curr] = first_out
                    #print("New {} -> New {} {}".format(\
                    #    prev, curr, self.layer_sizes[l - 1]))
            to_net['output'] = to_net['fc{}'.format(self.L + 1)]
            to_net['predicted'] = tf.cast(tf.argmax(tf.nn.softmax( \
                    to_net['output']), axis=-1), tf.int64)
            net = to_net
        elif self.from_arch == None and self.to_arch == 'fc':
            fc_decoder = self.fc_decoder()
            net = fc_decoder

        print( "Total number of variables used ", np.sum([v.get_shape().num_elements() for v in tf.trainable_variables()]))

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=net['output'], labels=self.labels))

        self.minimizer = tf.train.AdamOptimizer(1e-5).minimize(self.loss)
        self.correct = tf.equal(net['predicted'], self.raw_labels)
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
                batch_images = np.moveaxis(batch_images, 3, 0)
                feed_dict = {self.inputs: batch_images, self.raw_labels: batch_labels_raw}
                acc, loss, _ = self.sess.run([self.accuracy, self.loss, self.minimizer], \
                    feed_dict=feed_dict) 
                accs.append(acc)
                losses.append(loss)
            eprint("[{}] Accuracy: {}, Loss = {}".format(epoch, np.mean(accs), np.mean(losses)))

            if epoch % 2 == 0 and epoch != 0:
                test_X, test_y = input_loader.get_test_data()
                test_X = np.moveaxis(test_X, 3, 0)
                #test_X = test_X.reshape((32 * 32 * 3, -1)).T
                feed_dict = {self.inputs: test_X, self.raw_labels: test_y}
                acc = self.sess.run([self.accuracy], feed_dict=feed_dict)
                eprint("[{}] Test accuracy: {}".format(epoch, acc)) 

def main():
    prog = ProgressiveNetwork(None, "fc")
    prog.train(64)

if __name__=="__main__":
    main()
