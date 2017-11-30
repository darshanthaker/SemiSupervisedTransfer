"""
    Implementation adapted from https://github.com/rinuboney/ladder/
"""
import tensorflow as tf
import input_data
import math
import os
import csv
import sys
from pdb import set_trace
from tqdm import tqdm
from util import eprint


class LadderNetwork(object):


    def __init__(self):
        # TODO(dbthaker): Change this first to be size of data.
        self.layer_sizes = [784, 1000, 500, 250, 250, 250, 10]
        self.batch_size = 100
        self.L = len(self.layer_sizes) - 1  # number of layers
        self.num_examples = 60000
        self.num_epochs = 150
        self.num_labeled = 100
        self.starter_learning_rate = 0.02
        self.decay_after = 15  # epoch after which to begin learning rate decay
        self.num_iter = int((self.num_examples//self.batch_size) * self.num_epochs)  # number of loop iterations

        self.create_compute_graph()

    def bi(self, inits, size, name, scope="ladder_net"):
        with tf.variable_scope(scope):
            return tf.Variable(inits * tf.ones([size]), name=name)

    def wi(self, shape, name, scope="ladder_net"):
        with tf.variable_scope(scope):
            return tf.Variable(tf.random_normal(shape, name=name), name=name)

    def true_divide(self, var, shape, scope="ladder_net"):
        with tf.variable_scope(scope):
            return var / math.sqrt(shape[0])

    def batch_normalization(self, batch, mean=None, var=None):
        if mean is None or var is None:
            mean, var = tf.nn.moments(batch, axes=[0])
        return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))

    "batch normalize + update average mean and variance of layer l"
    def update_batch_normalization(self, batch, l):
        mean, var = tf.nn.moments(batch, axes=[0])
        self.assign_mean = self.running_mean[l-1].assign(mean)
        self.assign_var = self.running_var[l-1].assign(var)
        self.bn_assigns.append(self.ewma.apply([self.running_mean[l-1], self.running_var[l-1]]))
        with tf.control_dependencies([self.assign_mean, self.assign_var]):
            return (batch - mean) / tf.sqrt(var + 1e-10)

    def create_encoder(self, inputs, noise_std):
        h = inputs + tf.random_normal(tf.shape(inputs)) * noise_std  # add noise to input
        d = {}  # to store the pre-activation, activation, mean and variance for each layer

        # The data for labeled and unlabeled examples are stored separately
        d['labeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
        d['unlabeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
        d['labeled']['z'][0], d['unlabeled']['z'][0] = self.split_lu(h)
        for l in range(1, self.L+1):
            eprint("Layer ", l, ": ", self.layer_sizes[l-1], " -> ", self.layer_sizes[l])
            d['labeled']['h'][l-1], d['unlabeled']['h'][l-1] = self.split_lu(h)
            z_pre = tf.matmul(h, self.weights['W'][l-1])  # pre-activation
            z_pre_l, z_pre_u = self.split_lu(z_pre)  # split labeled and unlabeled examples

            m, v = tf.nn.moments(z_pre_u, axes=[0])

            # if training:
            def training_batch_norm():
                # Training batch normalization
                # batch normalization for labeled and unlabeled examples is performed separately
                if noise_std > 0:
                    # Corrupted encoder
                    # batch normalization + noise
                    z = self.join(self.batch_normalization(z_pre_l), self.batch_normalization(z_pre_u, m, v))
                    z += tf.random_normal(tf.shape(z_pre)) * noise_std
                else:
                    # Clean encoder
                    # batch normalization + update the average mean and variance using batch mean and variance of labeled examples
                    z = self.join(self.update_batch_normalization(z_pre_l, l), self.batch_normalization(z_pre_u, m, v))
                return z

            # else:
            def eval_batch_norm():
                # Evaluation batch normalization
                # obtain average mean and variance and use it to normalize the batch
                mean = self.ewma.average(self.running_mean[l-1])
                var = self.ewma.average(self.running_var[l-1])
                z = self.batch_normalization(z_pre, mean, var)
                # Instead of the above statement, the use of the following 2 statements containing a typo
                # consistently produces a 0.2% higher accuracy for unclear reasons.
                # m_l, v_l = tf.nn.moments(z_pre_l, axes=[0])
                # z = join(batch_normalization(z_pre_l, m_l, mean, var), batch_normalization(z_pre_u, mean, var))
                return z

            # perform batch normalization according to value of boolean "training" placeholder:
            z = tf.cond(self.training, training_batch_norm, eval_batch_norm)

            if l == self.L:
                # use softmax activation in output layer
                h = tf.nn.softmax(self.weights['gamma'][l-1] * (z + self.weights["beta"][l-1]))
            else:
                # use ReLU activation in hidden layers
                h = tf.nn.relu(z + self.weights["beta"][l-1])
            d['labeled']['z'][l], d['unlabeled']['z'][l] = self.split_lu(z)
            d['unlabeled']['m'][l], d['unlabeled']['v'][l] = m, v  # save mean and variance of unlabeled examples for decoding
        d['labeled']['h'][l], d['unlabeled']['h'][l] = self.split_lu(h)
        return h, d

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
        z_est = {}
        self.d_cost = []  # to store the denoising cost of all layers
        for l in range(self.L, -1, -1):
            eprint("Layer ", l, ": ", self.layer_sizes[l+1] if l+1 < len(self.layer_sizes) else None, " -> ", self.layer_sizes[l], ", denoising cost: ", self.denoising_cost[l])
            z, z_c = self.clean['unlabeled']['z'][l], self.corr['unlabeled']['z'][l]
            m, v = self.clean['unlabeled']['m'].get(l, 0), self.clean['unlabeled']['v'].get(l, 1-1e-10)
            if l == self.L:
                u = self.unlabeled(self.y_c)
            else:
                u = tf.matmul(z_est[l+1], self.weights['V'][l])
            u = self.batch_normalization(u)
            z_est[l] = self.g_gauss(z_c, u, self.layer_sizes[l])
            z_est_bn = (z_est[l] - m) / v
            # append the cost of this layer to d_cost
            self.d_cost.append((tf.reduce_mean(tf.reduce_sum(tf.square(z_est_bn - z), 1)) / self.layer_sizes[l]) * self.denoising_cost[l])

    def create_compute_graph(self):
        self.inputs = tf.placeholder(tf.float32, shape=(None, self.layer_sizes[0]))
        self.outputs = tf.placeholder(tf.float32)
        self.training = tf.placeholder(tf.bool)

        # Shapes of linear layers.
        shapes = list(zip(self.layer_sizes[:-1], self.layer_sizes[1:]))

        self.weights = {'W_raw': [self.wi(s, "W", scope="transfer_weights") \
                              for s in shapes],  # Encoder weights
                   'V': [self.wi(s[::-1], "V") for s in shapes],  # Decoder weights
                   # batch normalization parameter to shift the normalized value
                   'beta': [self.bi(0.0, self.layer_sizes[l+1], "beta", scope="transfer_weights") \
                            for l in range(self.L)],
                   # batch normalization parameter to scale the normalized value
                   'gamma': [self.bi(1.0, self.layer_sizes[l+1], "beta") for l in range(self.L)]}

        self.weights['W'] = [self.true_divide(w, s, scope="transfer_weights") \
                             for (w, s) in zip(self.weights['W_raw'], shapes)]

        self.noise_std = 0.3  # scaling factor for noise used in corrupted encoder

        # hyperparameters that denote the importance of each layer
        self.denoising_cost = [1000.0, 10.0, 0.10, 0.10, 0.10, 0.10, 0.10]

        self.join = lambda l, u: tf.concat([l, u], 0)
        self.labeled = lambda x: tf.slice(x, [0, 0], [self.batch_size, -1]) if x is not None else x
        self.unlabeled = lambda x: tf.slice(x, [self.batch_size, 0], [-1, -1]) if x is not None else x
        self.split_lu = lambda x: (self.labeled(x), self.unlabeled(x))

        self.ewma = tf.train.ExponentialMovingAverage(decay=0.99)  # to calculate the moving averages of mean and variance
        self.bn_assigns = []  # this list stores the updates to be made to average mean and variance

        # average mean and variance of all layers
        self.running_mean = [tf.Variable(tf.constant(0.0, shape=[l]), trainable=False) for l in self.layer_sizes[1:]]
        self.running_var = [tf.Variable(tf.constant(1.0, shape=[l]), trainable=False) for l in self.layer_sizes[1:]]

        
        eprint("=== Corrupted Encoder ===")
        self.y_c, self.corr = self.create_encoder(self.inputs, self.noise_std)

        eprint("=== Clean Encoder ===")
        self.y, self.clean = self.create_encoder(self.inputs, 0.0)  # 0.0 -> do not add noise

        eprint("=== Decoder ===")
        self.create_decoder()

        # calculate total unsupervised cost by adding the denoising cost of all layers
        self.u_cost = tf.add_n(self.d_cost)

        self.y_N = self.labeled(self.y_c)
        self.cost = -tf.reduce_mean(tf.reduce_sum(self.outputs*tf.log(self.y_N), 1))  # supervised cost
        self.loss = self.cost + self.u_cost  # total cost

        self.pred_cost = -tf.reduce_mean(tf.reduce_sum(self.outputs*tf.log(self.y), 1))  # cost used for prediction

        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.outputs, 1))  # no of correct predictions
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float")) * tf.constant(100.0)

        self.learning_rate = tf.Variable(self.starter_learning_rate, trainable=False)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # add the updates of batch normalization statistics to train_step
        bn_updates = tf.group(*self.bn_assigns)
        with tf.control_dependencies([self.train_step]):
            self.train_step = tf.group(bn_updates)

    def train(self):
        eprint("===  Loading Data ===")
        mnist = input_data.read_data_sets("MNIST_data", n_labeled=self.num_labeled, one_hot=True)

        saver = tf.train.Saver()
        var_dict = dict()
        for l in range(self.L):
            var_dict[self.weights['W_raw'][l].name[:-2]] = self.weights['W_raw'][l]
            var_dict[self.weights['beta'][l].name[:-2]] = self.weights['beta'][l]
            
        weights_saver = tf.train.Saver(var_dict)

        eprint("===  Starting Session ===")
        self.sess = tf.Session()

        i_iter = 0

        ckpt = tf.train.get_checkpoint_state('checkpoints/')  # get latest checkpoint (if any)
        if ckpt and ckpt.model_checkpoint_path:
            # if checkpoint exists, restore the parameters and set epoch_n and i_iter
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            epoch_n = int(ckpt.model_checkpoint_path.split('-')[1])
            i_iter = (epoch_n+1) * (self.num_examples//self.batch_size)
            eprint("Restored Epoch ", epoch_n)
        else:
            # no checkpoint exists. create checkpoints directory if it does not exist.
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
            if not os.path.exists('weight_checkpoints'):
                os.makedirs('weight_checkpoints')
            init = tf.global_variables_initializer()
            self.sess.run(init)

        eprint("=== Training ===")
        eprint("Initial Accuracy: ", self.sess.run(self.accuracy, feed_dict={self.inputs: mnist.test.images, self.outputs: mnist.test.labels, self.training: False}), "%")

        for i in range(i_iter, self.num_iter):
            images, labels = mnist.train.next_batch(self.batch_size)
            self.sess.run(self.train_step, feed_dict={self.inputs: images, self.outputs: labels, self.training: True})
            if i % 10 == 0:
                eprint("[{}] Test Accuracy: ".format(i), self.sess.run(self.accuracy, feed_dict={self.inputs: mnist.test.images, self.outputs: mnist.test.labels, self.training: False}), "%")
            if (i > 1) and ((i+1) % (self.num_iter//self.num_epochs) == 0):
                epoch_n = i//(self.num_examples//self.batch_size)
                if (epoch_n+1) >= self.decay_after:
                    # decay learning rate
                    # learning_rate = starter_learning_rate * ((num_epochs - epoch_n) / (num_epochs - decay_after))
                    ratio = 1.0 * (self.num_epochs - (epoch_n+1))  # epoch_n + 1 because learning rate is set for next epoch
                    ratio = max(0, ratio / (self.num_epochs - self.decay_after))
                    self.sess.run(self.learning_rate.assign(self.starter_learning_rate * ratio))
                saver.save(self.sess, 'checkpoints/model.ckpt', epoch_n)
                weights_saver.save(self.sess, "weight_checkpoints/weights.ckpt", epoch_n)
                # print "Epoch ", epoch_n, ", Accuracy: ", sess.run(accuracy, feed_dict={inputs: mnist.test.images, outputs:mnist.test.labels, training: False}), "%"
                with open('train_log', 'a') as train_log:
                    # write test accuracy to file "train_log"
                    train_log_w = csv.writer(train_log)
                    log_i = [epoch_n] + self.sess.run([self.accuracy], feed_dict={self.inputs: mnist.test.images, self.outputs: mnist.test.labels, self.training: False})
                    train_log_w.writerow(log_i)

        
        eprint("Final Accuracy: ", self.sess.run(self.accuracy, feed_dict={self.inputs: mnist.test.images, self.outputs: mnist.test.labels, self.training: False}), "%")
        self.sess.close()

def main():
    ladder = LadderNetwork()
    ladder.train()

if __name__=="__main__":
    main()
