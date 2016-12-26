#!/usr/bin/python3.5

"""
Use feed forward neural network to predict the border of the deformable object,
given the force detected at the finger and the finger position.

View logs with:
$ tensorboard --logdir=logs
"""

import sys
import numpy as np
import tensorflow as tf

from Util import print_msg


LOG_DIR = "logs"


param_suits = {
    'sponge_set_1': {
        'train': False,
        'file_train_data': 'data/pickles/sponge_set_1_track.npz',
        'file_force_data': 'data/original/sponge_centre_100.txt',
        'file_restore': 'data/pickles/sponge_ff.ckpt',
        'file_predictions': 'data/pickles/sponge_center_predictions.csv',
        'file_video': 'data/generated_videos/sponge_centre_100__filterless_segmented.avi',
        'hidden_neurons': 50,
        'learning_rate': (0.03, 0.005, 0.003),
        'momentum': (0.01, 0.003, 0.001),
        'train_epochs': (50, 4000, 6000),
        'force_norm': 3.5,
        'pixel_norm': 600.0
    },
    'sponge_set_2': {
        'train': False,
        'file_train_data': 'data/pickles/sponge_set_2_track.npz',
        'file_force_data': 'data/original/sponge_longside_100.txt',
        'file_restore': 'data/pickles/sponge_ff.ckpt',
        'file_predictions': 'data/pickles/sponge_longside_predictions.csv',
        'file_video': 'data/generated_videos/sponge_longside_100__filterless_segmented.avi',
        'force_norm': 3.5,
        'pixel_norm': 600.0
    },
    'sponge_set_3': {
        'train': False,
        'file_train_data': 'data/pickles/sponge_set_3_track.npz',
        'file_force_data': 'data/original/sponge_shortside_100.txt',
        'file_restore': 'data/pickles/sponge_ff.ckpt',
        'file_predictions': 'data/pickles/sponge_shortside_predictions.csv',
        'file_video': 'data/generated_videos/sponge_shortside_100__filterless_segmented.avi',
        'force_norm': 3.5,
        'pixel_norm': 600.0
    },
    'plasticine_set_1': {
        'train': True,
        'file_train_data': 'data/pickles/plasticine_set_1_track.npz',
        'file_force_data': 'data/original/plasticine_centre_100_below.txt',
        'file_restore': 'data/pickles/plasticine_ff.ckpt',
        'file_predictions': 'data/pickles/plasticine_center_predictions.csv',
        'file_video': 'data/generated_videos/a_plasticine_centre_100__filterless_segmented.avi',
        'hidden_neurons': 70,
        'learning_rate': (0.03, 0.03, 0.007, 0.003),
        'momentum': (0.03, 0.01, 0.005, 0.001),
        'train_epochs': (100, 4000, 10000, 10000),
        'force_norm': 3.5,
        'pixel_norm': 600.0
    },
    'plasticine_set_2': {
        'train': False,
        'file_train_data': 'data/pickles/plasticine_set_2_track.npz',
        'file_force_data': 'data/original/plasticine_longside_100_below.txt',
        'file_restore': 'data/pickles/plasticine_ff.ckpt',
        'file_predictions': 'data/pickles/plasticine_longside_predictions.csv',
        'file_video': 'data/generated_videos/plasticine_longside_100__filterless_segmented.avi',
        'hidden_neurons': 70,
        'force_norm': 3.5,
        'pixel_norm': 600.0
    },
    'plasticine_set_3': {
        'train': False,
        'file_train_data': 'data/pickles/plasticine_set_3_track.npz',
        'file_force_data': 'data/original/plasticine_shortside_100_below.txt',
        'file_restore': 'data/pickles/plasticine_ff.ckpt',
        'file_predictions': 'data/pickles/plasticine_shortside_predictions.csv',
        'file_video': 'data/generated_videos/plasticine_shortside_100__filterless_segmented.avi',
        'hidden_neurons': 70,
        'force_norm': 3.5,
        'pixel_norm': 600.0
    }
}


class FFDeformation:
    """ Neural network of three layers used to predict deformation from
    finger position and force.
    """
    def __init__(self, num_neurons, norm_constants, file_restore=None):
        """
        Creates variables for a three layer network
        :param num_neurons: Tuple of number of neurons per layer
        :param norm_constants: List of normalization constants for (x, y, force)
        """
        self.sess = sess = tf.InteractiveSession()
        with tf.name_scope('Input'):
            self.x = x = tf.placeholder(tf.float32, shape=[None, num_neurons[0]], name='X')
            norm_x = tf.constant(norm_constants, name='normX')
            norm_input = tf.div(x, norm_x)
            self.y_ = y_ = tf.placeholder(tf.float32, shape=[None, num_neurons[2]], name='y_')
            norm_y = tf.constant(norm_constants[0:2] * int(num_neurons[2]/2), name='normY')
            norm_desired_output = tf.div(y_, norm_y)
            tf.histogram_summary('Input/x', x)
            tf.histogram_summary('Input/normalized_x', norm_input)
            tf.histogram_summary('Input/y_', y_)
            tf.histogram_summary('Input/normalized_y_', norm_desired_output)

        with tf.name_scope('Hidden'):
            W1 = tf.Variable(tf.random_uniform([num_neurons[0], num_neurons[1]], -1.0, 1.0), name='W1')
            b1 = tf.Variable(tf.constant(0.1, shape=(num_neurons[1],)), name='b1')
            h = tf.nn.sigmoid(tf.matmul(norm_input, W1) + b1, name='h')
            tf.histogram_summary('Hidden/W1', W1)
            tf.histogram_summary('Hidden/b1', b1)
            tf.histogram_summary('Hidden/h', h)

        with tf.name_scope('Output'):
            W2 = tf.Variable(tf.random_uniform([num_neurons[1], num_neurons[2]], -1.0, 1.0), name='W2')
            b2 = tf.Variable(tf.constant(0.1, shape=(num_neurons[2],)), name='b2')
            self.y = y = tf.nn.sigmoid(tf.matmul(h, W2) + b2, name='y')
            self.out = tf.mul(y, norm_y)
            tf.histogram_summary('Output/W2', W2)
            tf.histogram_summary('Output/b2', b2)
            tf.histogram_summary('Output/y', y)
            tf.histogram_summary('Output/out', self.out)

        with tf.name_scope('Error'):
            self.error = tf.reduce_mean(tf.nn.l2_loss(tf.sub(y, norm_desired_output)), name='Error')
            #tf.summary.scalar("Error", self.error)
            self.error_summary = tf.scalar_summary("Error", self.error)

        # Merge all the summaries
        #self.merged = tf.summary.merge_all()
        self.merged = tf.merge_all_summaries()
        self.train_writer = tf.train.SummaryWriter(LOG_DIR + '/train', sess.graph)
        self.val_writer = tf.train.SummaryWriter(LOG_DIR + '/val', sess.graph)

        # Prepare for saving network state
        self.saver = tf.train.Saver()
        if file_restore is None:
            sess.run(tf.initialize_all_variables())
        else:
            self.saver.restore(self.sess, file_restore)
            print_msg("Model restored from ", file_restore)

        self.trained_cycles = 0

    def train(self, X, Y, X_val, Y_val, learning_rate, momentum, cycles):
        """
        Adjust weights using data in
        :param X: X = [(x,y,force),...]
        :param Y: Y = [(contour_points)]
        :param learning_rate: for learning algorithm (gradient descent or adam?)
        :return:
        """
        NUM_LOG_POINTS = min(1000, cycles)
        log_rate = max(int(cycles/NUM_LOG_POINTS), 5)

        temp = set(tf.all_variables())
        #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.error)
        #train_step = tf.train.AdagradOptimizer(learning_rate).minimize(self.error)
        train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(self.error)
        self.sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
        #train_step = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08,
        #                                    use_locking=False, name='Adam').minimize(self.error)
        #init_training = tf.cond(tf.not_equal(tf.report_uninitialized_variables(), None),
        #                        lambda: tf.initialize_variables(tf.report_uninitialized_variables()),
        #                        lambda: tf.no_op())
        #self.sess.run(tf.initialize_variables(tf.report_uninitialized_variables()))

        trained_cycles = self.trained_cycles
        for i in range(cycles):
            if i % log_rate == 0:
                summary, acc = self.sess.run([self.merged, train_step],
                                             feed_dict={self.x: X, self.y_: Y})
                self.train_writer.add_summary(summary, trained_cycles)
                summary, acc = self.sess.run([self.error_summary, self.error],
                                             feed_dict={self.x: X_val, self.y_: Y_val})
                self.val_writer.add_summary(summary, trained_cycles)
                if i % (log_rate * 100) == 0:
                    print_msg("\tAdvance ", i*100/cycles, '%')
            else:
                train_step.run(feed_dict={self.x: X, self.y_: Y}, session = self.sess)
            trained_cycles += 1
        self.trained_cycles = trained_cycles

    def evaluate(self, X, Y):
        """ Evaluates the performance of the net on given X and Y, with current
        values for the weights.
        """
        return self.error.eval(feed_dict={self.x: X, self.y_: Y}, session = self.sess)

    def feed_forward(self, X):
        """ Calculates the output of the network for data X. """
        return self.sess.run(self.out, feed_dict={self.x: X})

    def save(self, file_checkpoint):
        """ Saves a checkpoint of the network from which it can be restored. """
        self.saver.save(self.sess, file_checkpoint)

    def close(self):
        """ Closes tensorflow resources. """
        self.train_writer.close()
        self.val_writer.close()
        self.sess.close()


def train(param_suit):
    """ Creates and trains FF. """
    X, Y = load_data(param_suit['file_train_data'],
                     param_suit['file_force_data'])
    print_msg("Loaded ", type(X), X.shape, type(Y), Y.shape)

    # Randomly chose training and validation sets
    rand_ind = np.arange(len(X))
    np.random.shuffle(rand_ind)
    X_train = X[rand_ind[:70], :]
    Y_train = Y[rand_ind[:70], :]
    X_val = X[rand_ind[70:], :]
    Y_val = Y[rand_ind[70:], :]

    # Create and train or restore FFnn.
    nn = None
    num_neurons = [3, param_suit['hidden_neurons'], Y.shape[1]]
    if not param_suit['train'] and tf.gfile.Exists(param_suit['file_restore']):
        print_msg("Restoring trained neural network...")
        nn = FFDeformation(num_neurons, (param_suit['pixel_norm'],
                                         param_suit['pixel_norm'],
                                         param_suit['force_norm']), param_suit['file_restore'])

        b_save = input("Do you want to save the predicted contour? [y/n] ")
        if b_save == 'y':
            predictions = nn.feed_forward(X)
            np.savetxt(param_suit['file_predictions'], predictions)
            print_msg("Predictions saved.")
    else:
        print_msg("Creating feedforward neural network...", num_neurons)
        nn = FFDeformation(num_neurons, (param_suit['pixel_norm'],
                                         param_suit['pixel_norm'],
                                         param_suit['force_norm']))
        #print_msg("Testing initialization...")
        #print("Y = ", nn.feed_forward(X))

        cont = "y"
        error = nn.evaluate(X, Y)
        print_msg("Initial error = ", error)
        for learning_rate, momentum, train_epochs in zip(param_suit['learning_rate'],
                                                         param_suit['momentum'],
                                                         param_suit['train_epochs']):
            #cont = input("Do you want to continue? [y/n] ")
            #if cont == "n": break
            print_msg("Training...")
            nn.train(X_train, Y_train, X_val, Y_val, learning_rate, momentum, train_epochs)

            print_msg("Evaluating...")
            error = nn.evaluate(X, Y)
            print_msg("Error on training set = ", error)

        if cont != "n":
            b_save = input("Do you want to save the network? [y/n] ")
            if b_save == 'y':
                nn.save(param_suit['file_restore'])
                predictions = nn.feed_forward(X)
                np.savetxt(param_suit['file_predictions'], predictions)
                print_msg("Network and predictions saved.")
    nn.close()


def load_data(track_file, force_file):
    """ Loads X, Y matrices for trainning of the neural network. """
    npzfile = np.load(track_file)
    X = npzfile['X']
    Y = npzfile['Y']
    X2 = np.loadtxt(force_file)[:,3:4]  # Want Tz on column 3
    print_msg("Finger data shapes: ", X.shape, X2.shape)
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

    train(param_suit)
