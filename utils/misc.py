import numpy as np
import tensorflow as tf
import progressbar
import urllib.request

class Struct():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def update(self, other):
        if isinstance(other, dict):
            self.__dict__.update(other)
        elif isinstance(other, Struct):
            self.__dict__.update(other.__dict__)

    def updated(self, other):
        copy = Struct(**self.__dict__)
        copy.update(other)
        return copy

    def update_exclusive(self, other):
        if isinstance(other, dict):
            d = other
        elif isinstance(other, Struct):
            d = other.__dict__
        for x in d:
            if x not in self.__dict__:
                self.__dict__[x] = d[x]
        return self

    def __repr__(self):
        return ''.join(['Struct('] + ['{}: {}\n'.format(repr(x), repr(y)) for x, y in self.__dict__.items()]+[')'])


def binarize(x, rng):
    return (rng.rand(*x.shape) < x).astype(np.float32)

#helper function for displaying reconstructed images.
def tile_images(array, n_cols=None):
    if n_cols is None:
        n_cols = int(np.sqrt(array.shape[0]))
    n_rows = int(np.ceil(float(array.shape[0])/n_cols))

    def cell(i, j):
        ind = i*n_cols+j
        if i*n_cols+j < array.shape[0]:
            return array[ind]
        else:
            return np.zeros(array[0].shape)

    def row(i):
        return np.concatenate([cell(i, j) for j in range(n_cols)], axis=1)

    return np.concatenate([row(i) for i in range(n_rows)], axis=0)

def tf_repeat(x, k):
    return tf.reshape(tf.tile(x, tf.stack([1, k])), tf.stack([-1, tf.shape(x)[1]]))

def tf_log_mean_exp(x):
    m = tf.reduce_max(x, 1, keepdims=True)
    return m + tf.log(tf.reduce_mean(tf.exp(x - m), 1, keepdims=True))

def download(url, filename):
    print("Downloading MNIST dataset")
    pbar = progressbar.ProgressBar()

    def dlProgress(count, blockSize, totalSize):
        if pbar.maxval is None:
            pbar.maxval = totalSize
            pbar.start()

        pbar.update(min(count*blockSize, totalSize))

    urllib.request.urlretrieve(url, filename, reporthook=dlProgress)
    pbar.finish()
