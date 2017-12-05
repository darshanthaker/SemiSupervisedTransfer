import tensorflow as tf
import numpy as np
import math
import pickle
import argparse
from util import eprint
from alexnet import AlexNet
import util
from pdb import set_trace
import visualize

def conv_layer(input, num_outputs, kernel_size, stride=1, scope="conv", reuse=None, \
            trainable=True):
    return tf.contrib.layers.conv2d(input, num_outputs=num_outputs,
        kernel_size=kernel_size, stride=stride, weights_regularizer=tf.nn.l2_loss, \
        activation_fn=None, scope=scope, reuse=reuse, trainable=trainable)

def pool_layer(input, pool_type):
    if pool_type == 'max':
        return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), \
                padding='SAME')
    elif pool_type == 'avg':
        return tf.reduce_mean(input, axis=[1, 2])

def fc_layer(input, num_outputs, scope="fc", activation_fn=None, reuse=None, \
            trainable=True):
    return tf.contrib.layers.fully_connected(input, num_outputs, activation_fn=activation_fn, \
            weights_regularizer=tf.nn.l2_loss, scope=scope, reuse=reuse, trainable=trainable)

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
    return input_layer + noise

# Transfer only between two tasks currently.
class ProgressiveNetwork(object):

    def __init__(self, from_arch, to_arch, perturbation):
        self.from_arch = from_arch
        self.to_arch = to_arch
        self.perturbation = perturbation
        if 'fc' in self.from_arch or 'fc' in to_arch:
            self.layer_sizes = [784, 1000, 500, 250, 250, 250, 10]
            self.L = len(self.layer_sizes) - 1
        else:
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

    def load_ladder_weights(self, arch, trainable=False):
        sess = tf.Session()
        shapes = list(zip(self.layer_sizes[:-1], self.layer_sizes[1:]))
        self.weights = {'W_raw': [self.wi(s, "W", trainable=trainable) \
                                  for (i, s) in enumerate(shapes)], \
                        'beta': [self.bi(0.0, self.layer_sizes[l + 1], \
                                'beta', trainable=trainable) for l in range(self.L)]}

        saver = tf.train.Saver()
        if arch == 'ladder':
            fm = 'weight_checkpoints'
        else:
            fm = 'baseline_checkpoints'
        ckpt = tf.train.get_checkpoint_state(fm)
        if ckpt and ckpt.model_checkpoint_path:
            #set_trace()
            #checkpoint_path = '{}/model.ckpt-9900'.format(fm)
            #checkpoint_path = '{}/weights.ckpt-2'.format(fm)
            checkpoint_path = ckpt.model_checkpoint_path
            saver.restore(sess, checkpoint_path)
            epoch_n = int(checkpoint_path.split('-')[1])
            eprint("Restored Epoch ", epoch_n)
        eprint("Restored weights from file {}".format(fm))

        self.weights['W'] = [self.true_divide(w, s, scope="transfer_weights") \
                             for (w, s) in zip(self.weights['W_raw'], shapes)]

    def fc_decoder(self, pretrain=False):
        net = {}
        self.gs_inputs = tf.image.rgb_to_grayscale(self.inputs)
        self.res_inputs = tf.image.resize_images(self.gs_inputs, (28, 28))
        net['fc0'] = tf.contrib.layers.flatten(self.res_inputs)

        # Shapes of linear layers.
        shapes = list(zip(self.layer_sizes[:-1], self.layer_sizes[1:]))
        if pretrain:
            self.weights['gamma'] = [self.bi(1.0, self.layer_sizes[l+1], "beta", trainable=True) for l in range(self.L)]
        else:
            self.weights = {'W_raw': [self.wi(s, "W", trainable=True) \
                                      for (i, s) in enumerate(shapes)], \
                            'beta': [self.bi(0.0, self.layer_sizes[l + 1], \
                                    'beta', trainable=True) for l in range(self.L)],
                           # batch normalization parameter to scale the normalized value
                           'gamma': [self.bi(1.0, self.layer_sizes[l+1], "beta", trainable=True) for l in range(self.L)]}
            self.weights['W'] = [self.true_divide(w, s, scope="transfer_weights") \
                                 for (w, s) in zip(self.weights['W_raw'], shapes)]

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

        return net

    def create_compute_graph(self):
        self.inputs = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
        self.raw_labels = tf.placeholder(tf.int64, shape=(None))
        self.labels = tf.one_hot(self.raw_labels, self.to_num_classes)
        self.src_noise_levels = tf.placeholder(tf.float32, shape=(self.L + 1))
        self.target_noise_levels = tf.placeholder(tf.float32, shape=(self.L + 1))

        if self.from_arch == 'ladder' and self.to_arch == 'alex':
            self.load_ladder_weights()
            alexnet_class = AlexNet()
            alexnet = alexnet_class.feed_forward(self.inputs)

            with tf.variable_scope('progressive_net'):
                alexnet['pool_3'] = tf.contrib.layers.flatten(alexnet['pool_3'])

                # TODO(dbthaker): Change activation fn
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
        elif (self.from_arch == 'ladder' or self.from_arch == 'baseline') \
                and self.to_arch == 'fc':
            self.gs_inputs = tf.image.rgb_to_grayscale(self.inputs)
            self.res_inputs = tf.image.resize_images(self.gs_inputs, (28, 28))
            self.load_ladder_weights(self.from_arch)
            from_net = {}
            to_net = {}
            from_net['fc0'] = tf.contrib.layers.flatten(self.res_inputs)
            from_net['fc0'] = gaussian_noise_layer(from_net['fc0'], self.src_noise_levels[0]) 
            to_net['fc0'] = tf.contrib.layers.flatten(self.res_inputs)
            to_net['fc0'] = gaussian_noise_layer(to_net['fc0'], self.target_noise_levels[0]) 

            for l in range(1, self.L + 2):
                prev = 'fc{}'.format(l - 1)
                curr = 'fc{}'.format(l)

                if l != self.L + 1:
                    from_net[curr] = tf.matmul(from_net[prev], self.weights['W'][l - 1])
                    from_net[curr] = tf.layers.batch_normalization(from_net[curr], training=False)
                    from_net[curr] = tf.nn.relu(from_net[curr] + self.weights['beta'][l - 1])
                    from_net[curr] = gaussian_noise_layer(from_net[curr], self.src_noise_levels[l - 1])

                first_out = fc_layer(to_net[prev], self.layer_sizes[l - 1], \
                    scope="first_fc{}".format(l), activation_fn=tf.nn.relu)
                #scale = tf.Variable(tf.random_normal([1], stddev=0.5))
                #first_out = scale * first_out
                print("Ladder {} -> Ladder {}".format(prev, curr))
                if l != 1:
                    second_out = fc_layer(from_net[prev], self.layer_sizes[l - 1], \
                        scope="second_fc{}".format(l), activation_fn=tf.nn.relu)
                    to_net[curr] = first_out + second_out
                    to_net[curr] = gaussian_noise_layer(to_net[curr], \
                        self.target_noise_levels[l - 1])
                    print("Ladder {} -> New {} {} + New {} -> New {} {}".format(\
                        prev, curr, self.layer_sizes[l - 1], \
                        prev, curr, self.layer_sizes[l - 1]))
                else:
                    to_net[curr] = first_out
                    to_net[curr] = gaussian_noise_layer(to_net[curr], \
                        self.target_noise_levels[l - 1])
                    print("New {} -> New {} {}".format(\
                        prev, curr, self.layer_sizes[l - 1]))
            to_net['output'] = to_net['fc{}'.format(self.L + 1)]
            to_net['predicted'] = tf.cast(tf.argmax(tf.nn.softmax( \
                    to_net['output']), axis=-1), tf.int64)
            net = to_net
            if self.from_arch == 'ladder':
                bounds = [70000]
                values = [1e-4, 1e-5]
            else:
                bounds = [70000]
                values = [1e-4, 1e-5]
            self.step_op = tf.Variable(0, name='step', trainable=False)
            self.lr = tf.train.piecewise_constant(self.step_op, bounds, values)
        elif (self.from_arch == 'ladder' or self.from_arch == 'baseline') \
                and self.to_arch == 'pre_fc':
            self.load_ladder_weights(self.from_arch, trainable=True)
            net = self.fc_decoder()
            bounds = [70000]
            values = [1e-4, 1e-5]
            self.step_op = tf.Variable(0, name='step', trainable=False)
            self.lr = tf.train.piecewise_constant(self.step_op, bounds, values)
        elif self.from_arch == "None" and self.to_arch == 'fc':
            fc_decoder = self.fc_decoder()
            net = fc_decoder
            bounds = [70000]
            values = [1e-4, 1e-5]
            self.step_op = tf.Variable(0, name='step', trainable=False)
            self.lr = tf.train.piecewise_constant(self.step_op, bounds, values)
        elif self.from_arch == 'conv_ladder' and self.to_arch == 'conv':
            self.gs_inputs = tf.image.rgb_to_grayscale(self.inputs)
            self.res_inputs = tf.image.resize_images(self.gs_inputs, (28, 28))

            net = {}
            net['0'] = self.res_inputs

            self.weights = {'beta': {k : self.bi(0.0, v[0], "beta", scope="transfer_weights") \
                                 for (k, v) in self.conv_params.items()}}
            # Hack: Hardcode last beta weight for fully connected + softmax at end.
            self.weights['beta'][8] = self.bi(0.0, 10, "beta", scope="transfer_weights")

            for (l, layer_type) in enumerate(self.layers):
                if l == 0:
                    prev = '0'
                else:
                    prev = "{}{}".format(self.layers[l - 1], l - 1)
                curr = "{}{}".format(self.layers[l], l)
                if layer_type == 'conv':
                    net[curr] = conv_layer(net[prev], self.conv_params[l][0], \
                                       self.conv_params[l][1], \
                                       scope="conv{}".format(l), trainable=False)
                elif layer_type == 'maxpool':
                    net[curr] = pool_layer(net[prev], 'max')
                elif layer_type == 'avgpool':
                    net[curr] = pool_layer(net[prev], 'avg')
                elif layer_type == 'fc':
                    set_trace()
                    net[prev] = tf.contrib.layers.flatten(net[prev])
                    net[curr] = fc_layer(net[prev], self.fc_params[l], \
                            scope="fc{}".format(l), trainable=False)

                if layer_type == 'conv':
                    net[curr] = tf.nn.relu(net[curr] + self.weights['beta'][l])
            sess = tf.Session()
            saver = tf.train.Saver()
            fm = 'conv_checkpoints'
            ckpt = tf.train.get_checkpoint_state(fm)
            if ckpt and ckpt.model_checkpoint_path:
                set_trace()
                checkpoint_path = ckpt.model_checkpoint_path
                saver.restore(sess, checkpoint_path)
                epoch_n = int(checkpoint_path.split('-')[1])
                eprint("Restored Epoch ", epoch_n)
            eprint("Restored weights from file {}".format(fm))

            from_net = net
            to_net = {}

            for (l, layer_type) in enumerate(self.layers):
                if l == 0:
                   prev = '0'
                else:
                    prev= '{}{}'.format(self.layers[l - 1], l - 1)
                curr = '{}{}'.format(self.layers[l], l)           

                if layer_type == 'conv':
                    first_out = conv_layer(from_net[prev], self.conv_params[l][0], \
                                      self.conv_params[l][1], \
                                      scope="first_conv{}".format(l))
                    second_out = conv_layer(to_net[prev], self.conv_params[l][0], \
                                      self.conv_params[l][1], \
                                      scope="sec_conv{}".format(l))
                    to_net[curr] = first_out + second_out
                elif layer_type == 'maxpool':
                    to_net[curr] = pool_layer(to_net[prev], 'max')
                elif layer_type == 'avgpool':
                    net[curr] = pool_layer(to_net[prev], 'avg')
                elif layer_type == 'fc':
                    net[prev] = tf.contrib.layers.flatten(net[prev])
                    net[curr] = fc_layer(net[prev], self.fc_params[l], scope="fc{}".format(l))

                if layer_type == 'conv':
                    net[curr] = tf.nn.relu(net[curr] + self.weights['beta'][l])

        elif self.from_arch == "None" and self.to_arch == 'conv':
            self.gs_inputs = tf.image.rgb_to_grayscale(self.inputs)
            self.res_inputs = tf.image.resize_images(self.gs_inputs, (28, 28))
            net = {}
            net['0'] = self.res_inputs

            self.weights = {'W_raw': [self.wi((10, 10), "W", scope="transfer_weights")], \
                        'beta': {k : self.bi(0.0, v[0], "beta", scope="transfer_weights") \
                                 for (k, v) in self.conv_params.items()},
                        'gamma': {8: self.bi(1.0, 10, "gamma")}}
            # Hack: Hardcode last beta weight for fully connected + softmax at end.
            self.weights['beta'][8] = self.bi(0.0, 10, "beta", scope="transfer_weights")

            for (l, layer_type) in enumerate(self.layers):
                if l == 0:
                    prev = '0'
                else:
                    prev = "{}{}".format(self.layers[l - 1], l - 1)
                curr = "{}{}".format(self.layers[l], l)
                if layer_type == 'conv':
                    net[curr] = conv_layer(net[prev], self.conv_params[l][0], \
                                       self.conv_params[l][1], \
                                       scope="conv{}".format(l))
                elif layer_type == 'maxpool':
                    net[curr] = pool_layer(net[prev], 'max')
                elif layer_type == 'avgpool':
                    net[curr] = pool_layer(net[prev], 'avg')
                elif layer_type == 'fc':
                    net[prev] = tf.contrib.layers.flatten(net[prev])
                    net[curr] = fc_layer(net[prev], self.fc_params[l], scope="fc{}".format(l))

                if layer_type == 'conv':
                    net[curr] = tf.nn.relu(net[curr] + self.weights['beta'][l])
                elif layer_type == 'fc':
                    net[curr] = tf.nn.softmax(self.weights['gamma'][l] * (net[curr] + \
                        self.weights['beta'][l]))
            net['output'] = net['fc{}'.format(self.L)]
            net['predicted'] = tf.cast(tf.argmax( \
                    net['output'], axis=-1), tf.int64)
            bounds = [15000]
            values = [1e-4, 1e-5]
            self.step_op = tf.Variable(0, name='step', trainable=False)
            self.lr = tf.train.piecewise_constant(self.step_op, bounds, values)

        eprint("{} architecture -> {} architecture".format(self.from_arch, self.to_arch))
        eprint( "Total number of variables used ", np.sum([v.get_shape().num_elements() for v in tf.trainable_variables()]))
        eprint("Learning rate: {}".format(self.lr))

        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=net['output'], labels=self.labels))
        self.loss = self.loss + 1e-6 * reg_loss

        self.minimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, \
                global_step=self.step_op)
        self.correct = tf.equal(net['predicted'], self.raw_labels)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))

    def train(self, batch_size):
        dir_name = '{}_{}_plots'.format(self.from_arch, self.to_arch)
        util.mkdir(dir_name)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
        input_loader = self.from_input_loader

        zero_noise_levels = [0] * (self.L + 1) 
        all_accs = list()
        all_losses = list()
        test_X, test_y = input_loader.get_test_data()
        test_X = np.moveaxis(test_X, 3, 0)
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
                feed_dict = {self.inputs: batch_images, self.raw_labels: batch_labels_raw, \
                             self.src_noise_levels: zero_noise_levels, \
                             self.target_noise_levels: zero_noise_levels}
                acc, loss, lr, _ = self.sess.run([self.accuracy, self.loss, self.lr, self.minimizer], \
                    feed_dict=feed_dict) 
                accs.append(acc)
                losses.append(loss)
            eprint("[{}] LR: {}, Accuracy: {}, Loss = {}".format(epoch, lr, np.mean(accs), np.mean(losses)))
            all_losses.append(np.mean(losses))

            if epoch % 1 == 0 and epoch != 0:
                #test_X = test_X.reshape((32 * 32 * 3, -1)).T
                feed_dict = {self.inputs: test_X, self.raw_labels: test_y, \
                        self.src_noise_levels: zero_noise_levels, \
                        self.target_noise_levels: zero_noise_levels}
                acc = self.sess.run([self.accuracy], feed_dict=feed_dict)
                all_accs.append(acc)
                eprint("[{}] Test accuracy: {}".format(epoch, acc)) 

        if not self.perturbation:
            util.serialize(all_accs, dir_name, 'all_accs')
            util.serialize(all_losses, dir_name, 'all_losses')
        else:
            best_acc = max(all_accs)
            best_acc = best_acc[0]
            eprint("Best accuracy: {}".format(best_acc))
            #threshold = min(0.5, best_acc)
            threshold = 0.5
            noise_levels = range(0, 400, 3)
            columns = ['src', 'target']
            noise_precision = dict()
            for c in columns:
                noise_precision[c] = dict()
                for l in range(self.L + 1):
                    done = False
                    for noise in noise_levels:
                        if done:
                            continue
                        src_noise_levels = [0] * (self.L + 1)
                        target_noise_levels = [0] * (self.L + 1)
                        if c == 'src':
                            src_noise_levels[l] = noise
                        elif c == 'target':
                            target_noise_levels[l] = noise
                        feed_dict = {self.inputs: test_X, self.raw_labels: test_y, \
                            self.src_noise_levels: src_noise_levels, \
                            self.target_noise_levels: target_noise_levels}
                        acc = self.sess.run([self.accuracy], feed_dict=feed_dict)
                        acc = acc[0]
                        #eprint("Column {}, Layer {}: noise precision: {}, acc: {}".format( \
                        #    c, l, noise, acc))
                        if best_acc * threshold >= acc and not done:
                            done = True
                            noise_precision[c][l] = noise
                            eprint("Column {}, Layer {}: noise precision: {}".format( \
                                c, l, noise))
            final_aps = dict()
            for c in columns:
                final_aps[c] = dict()
                for l in range(self.L + 1):
                    if c == 'src':
                        flip = 'target'
                    else:
                        flip = 'src'
                    final_aps[c][l] = noise_precision[c][l] / (noise_precision[c][l] + \
                        noise_precision[flip][l])
            with open(dir_name + '/final_aps', 'wb') as fp:
                pickle.dump(final_aps, fp)

def main(from_arch, to_arch, perturbation):
    prog = ProgressiveNetwork(from_arch, to_arch, perturbation)
    prog.train(64)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Progressive Network')
    parser.add_argument('--from_arch', help="")
    parser.add_argument('--to_arch', help="")
    parser.add_argument('--perturbation', help="", type=util.str2bool)
    args = parser.parse_args()
    main(args.from_arch, args.to_arch, args.perturbation)
