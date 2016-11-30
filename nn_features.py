import os
import numpy as np
from tfnn import Neural_Net
from mfcc import create_feature, mel_spectrogram

class MLPFeatureGenerator(object):
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

    def get_input_features(self, dir0, dir1, threshold=0.6, tmax=120):
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


def save_nn_output(rootdir, model, sess, nframe, threshold, subdirs=None, verbose=False):
    if not subdirs:
        subdirs = ['sober', 'drunk']
    for sdir in subdirs:
        dirpath = os.path.join(rootdir, sdir)
        newpath = os.path.join(rootdir, 'nn_features', sdir)
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        files = [f for f in os.listdir(dirpath) if f.endswith(".wav")]
        for filename in files:
            if verbose:
                print filename
            X = mel_spectrogram(os.path.join(dirpath, filename), nframe, threshold, tmax=10000)
            nnf = np.hstack((X, model.hidden_out(sess, X, 1), model.hidden_out(sess, X, 2)))
            np.savetxt(os.path.join(newpath, filename[:-4]+".csv"), nnf, fmt="%.4f", delimiter=",")
