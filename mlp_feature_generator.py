from mfcc import create_feature
import tensorflow as tf
import numpy as np


class MLPFeatureGenerator(object):
    """
    MLPFeatureGenerator is a class which convert timeseries to mel-log spectrogram,
    then gives outputs of trained 2-hidden-layer neural net in each layer
    """
    def __init__(self, nframe=9, h1=32, h2=8, reg=0.01):
        """
        Arg
        :param nframe: the number of frames using for NN input
        :param h1: the number of 1st hidden units
        :param h2: the number of 2nd hidden units
        :param reg: regularization parameter
        """
        self.model = Neural_Net([40*nframe, h1, h2, 1], regularization=reg)
        self.nframe = nframe

    def get_input_features(self, dir0, dir1, threshold=0.06, tmax=120):
        """
        get input features (mel-log spectrogram) for NN model
        Arg
        :param dir0: dirpath to the dataset with label 0
        :param dir1: dirpath to the dataset with label 1
        :param threshold: threshold for removing low-energy frames
        :param tmax: the maximum time signal to convert per a file
                    (120 means that you will use 120 seconds)
        """
        X, y = create_feature(dir0, dir1, self.nframe, threshold, tmax=tmax)
        return X, y

    def fit(self, X, y, epochs=50, lr = 0.01):
        self.sess = self.model.do_training(X, y, epochs=epochs, learning_rate=lr)

    def evaluate(self, X, y):
        return self.model.do_eval(self.sess, X, y)

    def convert(self, X):
        hout1 = self.model.hidden_out(self.sess, X, 1)
        hout2 = self.model.hidden_out(self.sess, X, 2)
        _, p = self.model.predict(self.sess, X)
        return hout1, hout2, p



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class Neural_Net(object):
    def __init__(self, layer_size, regularization=0):
        W = {}
        b = {}
        self.a = {}
        L = len(layer_size) - 1
        self.x_placeholder = tf.placeholder(tf.float32, [None, layer_size[0]])
        self.y_placeholder = tf.placeholder(tf.float32, [None, layer_size[-1]])
        self.a[0] = self.x_placeholder
        for l in range(1, L):
            W[l] = weight_variable([layer_size[l - 1], layer_size[l]])
            b[l] = bias_variable([layer_size[l]])
            self.a[l] = tf.nn.relu(tf.matmul(self.a[l - 1], W[l]) + b[l])

        W[L] = weight_variable([layer_size[L - 1], layer_size[L]])
        b[L] = bias_variable([layer_size[L]])
        self.logits = tf.matmul(self.a[L - 1], W[L]) + b[L]

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.logits, self.y_placeholder))
        for l in range(1, L):
            self.loss += regularization * (tf.nn.l2_loss(W[l]) + tf.nn.l2_loss(b[l]))

    def do_training(self, x_train, y_train, epochs=30, batch_size=100,
                   learning_rate=0.5, verbose=False):
        N = x_train.shape[0]
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        for i in xrange(epochs):
            p = np.random.permutation(N)
            x_train = x_train[p]
            y_train = y_train[p]
            batches = [(x_train[k:k+batch_size], y_train[k:k+batch_size]) for k in xrange(0, N, batch_size)]
            for xbatch, ybatch in batches:
                sess.run(train_step, feed_dict={self.x_placeholder: xbatch, self.y_placeholder: ybatch})
            if verbose:
                train_loss = sess.run(self.loss, feed_dict={self.x_placeholder: xbatch, self.y_placeholder: ybatch})
                print("epoch %d, loss %g" % (i, train_loss))
        return sess

    def do_eval(self, sess, x_test, y_test):
        correct_prediction = tf.equal(tf.cast(tf.greater(self.logits, 0), tf.float32), self.y_placeholder)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return sess.run(accuracy, feed_dict={self.x_placeholder: x_test, self.y_placeholder: y_test})

    def predict(self, sess, x):
        y_pred = tf.greater(self.logits, 0)
        p = tf.sigmoid(self.logits)
        return sess.run([y_pred, p], feed_dict={self.x_placeholder: x})

    def hidden_out(self, sess, x, l):
        return sess.run(self.a[l], feed_dict={self.x_placeholder: x})
