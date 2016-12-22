#!/usr/bin/python3.5

"""
Use feed forward neural network to predict the border of the deformable object,
given the force detected at the finger and the finger position.

View logs with:
$ tensorboard --logdir=logs
"""

import sys
import numpy as np
from sklearn.preprocessing import normalize
import tensorflow as tf

LOG_DIR = "logs"

param_suits = {
    'sponge_set_1': {
        'file_train_data': 'data/pickles/sponge_set_1_track.npz',
        'file_force_data': 'data/original/sponge_centre_100.txt',
        'learning_rate': 0.001,
        'train_epochs':10000,
        'roi_shape': (600, 500)
    }
}


def print_msg(*msg):
    """ Prints message to stdout with color. """
    colour_format = '0;36'
    print('\x1b[%sm%s\x1b[0m' % (colour_format, " ".join([m if isinstance(m, str) else str(m) for m in msg])))


class FFDeformation:
    """ Neural network of three layers used to predict deformation from
    finger position and force.
    """
    def __init__(self, num_neurons):
        """
        Creates variables for a three layer network
        :param num_neurons: Tuple of number of neurons per layer
        """
        self.sess = sess = tf.InteractiveSession()
        with tf.name_scope('Input'):
            self.x = x = tf.placeholder(tf.float32, shape=[None, num_neurons[0]], name='X')
            self.y_ = y_ = tf.placeholder(tf.float32, shape=[None, num_neurons[2]], name='Y_')
            tf.histogram_summary('Input/x', x)
            tf.histogram_summary('Input/y_', y_)

        with tf.name_scope('Hidden'):
            W1 = tf.Variable(tf.random_uniform([num_neurons[0], num_neurons[1]], -1.0, 1.0), name='W1')
            b1 = tf.Variable(tf.constant(0.1, shape=(num_neurons[1],)), name='b1')
            h = tf.nn.relu(tf.matmul(x, W1) + b1, name='h')
            tf.histogram_summary('Hidden/W1', W1)
            tf.histogram_summary('Hidden/b1', b1)
        with tf.name_scope('Output'):
            W2 = tf.Variable(tf.random_uniform([num_neurons[1], num_neurons[2]], -1.0, 1.0), name='W2')
            b2 = tf.Variable(tf.constant(0.1, shape=(num_neurons[2],)), name='b2')
            self.y = y = tf.nn.relu(tf.add(tf.matmul(h, W2), b2), name='Y')
            tf.histogram_summary('Output/W2', W2)
            tf.histogram_summary('Output/b2', b2)
            #tf.histogram_summary('Output' + '/y', y)
        with tf.name_scope('Error'):
            self.error = tf.reduce_mean(tf.nn.l2_loss(tf.sub(y, y_)), name='Error')
            #tf.summary.scalar("Error", self.error)
            tf.scalar_summary("Error", self.error)
        # Merge all the summaries
        #self.merged = tf.summary.merge_all()
        self.merged = tf.merge_all_summaries()
        self.train_writer = tf.train.SummaryWriter(LOG_DIR + '/train', sess.graph)
        #self.train_step = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam').minimize(self.error)
        #self.train_step = tf.train.MomentumOptimizer(0.01, 0.5).minimize(self.error)

        sess.run(tf.initialize_all_variables())

    def train(self, X, Y, X_val, Y_val, learning_rate, cycles):
        """
        Adjust weights using data in
        :param X: X = [(x,y,force),...]
        :param Y: Y = [(contour_points)]
        :param learning_rate: for learning algorithm (gradient descent or adam?)
        :return:
        """
        NUM_LOG_POINTS = min(1000, cycles)
        log_rate = max(int(cycles/NUM_LOG_POINTS), 5)

        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.error)
        #train_step = self.train_step
        #train_step = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08,
        #                                    use_locking=False, name='Adam').minimize(self.error)
        #init_training = tf.cond(tf.not_equal(tf.report_uninitialized_variables(), None),
        #                        lambda: tf.initialize_variables(tf.report_uninitialized_variables()),
        #                        lambda: tf.no_op())
        #self.sess.run(tf.initialize_variables(tf.report_uninitialized_variables()))
        for i in range(cycles):
            if i % log_rate == 0:
                summary, acc = self.sess.run([self.merged, train_step],
                                             feed_dict={self.x: X, self.y_: Y})
                self.train_writer.add_summary(summary, i)
                if i % (log_rate * 100) == 0:
                    print_msg("\tAdvance ", i*100/cycles, '%')
            train_step.run(feed_dict={self.x: X, self.y_: Y}, session = self.sess)
        self.train_writer.close()

    def evaluate(self, X, Y):
        """ Evaluates the performance of the net on given X and Y, with current
        values for the weights.
        """
        return self.error.eval(feed_dict={self.x: X, self.y_: Y}, session = self.sess)

    def feed_forward(self, X):
        """ Calculates the output of the network for data X. """
        return self.sess.run(self.y, feed_dict={self.x: X})


def train(param_suit):
    """ Creates and trains FF. """
    X, Y = load_data(param_suit['file_train_data'],
                     param_suit['file_force_data'])
    print_msg("Loaded ", type(X), X.shape, type(Y), Y.shape)
    # sys.exit(-1)
    num_neurons = [3, 50, Y.shape[1]]
    print_msg("Creating feedforward neural network...", num_neurons)
    nn = FFDeformation(num_neurons)
    #print_msg("Testing initialization...")
    #print("Y = ", nn.feed_forward(X))
    print_msg("Training...")
    nn.train(X, Y, param_suit['learning_rate'], param_suit['train_epochs'])
    print_msg("Evaluating...")
    error = nn.evaluate(X, Y)
    print_msg("Error on training set = ", error)
    return nn


def load_data(track_file, force_file):
    """ Loads X, Y matrices for trainning of the neural network. """
    npzfile = np.load(track_file)
    X = npzfile['X']
    Y = npzfile['Y']
    X2 = np.loadtxt(force_file)[:,3:4]  # Want Tz on column 3
    X = np.hstack((X,X2))
    return X, Y


def print_instructions():
    """ Prints usage instructions. """
    print("Use: " + sys.argv[0] + " <set_of_paramteres>\nOptions:")
    for key in param_suits.keys():
        print('\t', key)


if __name__ == '__main__':
    nargs = len(sys.argv)
    if nargs != 2 or sys.argv[1] not in param_suits:
        print_instructions()
        sys.exit(1)
    param_suit = param_suits[sys.argv[1]]

    # Prepare directory for log files
    if tf.gfile.Exists(LOG_DIR):
        tf.gfile.DeleteRecursively(LOG_DIR)
    tf.gfile.MakeDirs(LOG_DIR)

    nn = train(param_suit)
