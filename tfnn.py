import tensorflow as tf
import numpy as np

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

def save_nn_output(rootdir, model, sess, nframe, threshold, subdirs= None):
    if not subdirs:
        subdirs = ['train', 'test', 'unused']
    for sdir in subdirs:
        dirpath = os.path.join(rootdir, sdir)
        files = [f for f in os.listdir(dirpath) if f.endswith(".wav")]
        for filename in files:
            X = mel_spectrogram(os.path.join(dirpath, filename), nframe, threshold, twin=25, tover=10, nceps=0)
            nnf = np.hstack((X, model.hidden_out(sess, X, 1), model.hidden_out(sess, X, 2)))
            newpath = os.path.join(rootdir, 'nn_features')
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            np.savetxt(os.path.join(newpath, filename[:-4]+".csv"),
                       nnf, fmt="%.4f", delimiter=",")
