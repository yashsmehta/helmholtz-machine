import os
import pickle
import gzip
import numpy as np
import utils.misc as misc
import utils.config as config


class MNIST():
    def __init__(self, binary=False):
        self.name = 'MNIST'
        self.url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        self.directory = os.path.join(config.DATADIR, self.name)
        self.filename = os.path.join(self.directory, 'mnist.pkl.gz')
        self.binary = binary
        self.train_rng = np.random.RandomState(283)

        self._ensure_file_is_on_disk()
        self.data = {}
        with gzip.open(self.filename, 'rb') as f:
            self.data['train'], self.data['valid'], self.data['test'] \
                = pickle.load(f, encoding="latin1")

    def _ensure_file_is_on_disk(self):
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)
        if not os.path.exists(self.filename):
            misc.download(self.url, self.filename)

    def minibatch(self, subdataset, indices, rng):
        data, labels = self.data[subdataset]
        data, labels = data[indices], labels[indices]
        if self.binary:
            data = misc.binarize(data, rng)
        return data, labels

    def get_random_minibatch(self, subdataset, minibatch_size, rng):
        indices = rng.randint(self.data[subdataset][0].shape[0], size=(minibatch_size,))
        return self.minibatch(subdataset, indices, rng)

    def random_minibatches(self, subdataset, minibatch_size, num_minibatches, rng=None):
        if subdataset != 'train' and rng is not None:
            print('Trying to access a stream of random minibatches from valid/test set will affect training rng!')
        if rng is None:
            rng = self.train_rng
        for _ in range(num_minibatches):
            yield self.get_random_minibatch(subdataset, minibatch_size, rng)

    def all_minibatches(self, subdataset, minibatch_size, rng, max_num_minibatches=None):
        num_minibatches = 1 + (self.data[subdataset][0].shape[0] - 1) // minibatch_size
        if max_num_minibatches is not None and max_num_minibatches < num_minibatches:
            num_minibatches = max_num_minibatches

        for i in range(num_minibatches):
            indices = slice(i * minibatch_size, (i + 1) * minibatch_size)
            yield self.minibatch(subdataset, indices, rng)

    def reshape_for_display(self, x):
        return x.reshape((-1, 28, 28))

    def get_data_dim(self):
        return 28*28
