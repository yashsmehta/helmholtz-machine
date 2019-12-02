import numpy as np
import tensorflow as tf
from collections import OrderedDict
from utils import Struct

#initializers
def Glorot(shape, scale=1.):
    scale = scale * np.sqrt(6.0 / np.sum(shape))
    return tf.Variable(tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=np.float32))

def ZeroBias(len):
    return tf.Variable(np.zeros((len,), dtype=np.float32))

#non-linearities
class NL():
    def __init__(self, nl):
        self.nl = nl
        self.params = []
        self.cache = {}

    def __call__(self, x, **kwargs):
        if x not in self.cache:
            y = self.nl(x)
            self.cache[x] = Struct(y=y)
        return self.cache[x]

    def to_short_str(self):
        to_str = {tf.nn.relu: 'relu',
                  tf.nn.sigmoid: 'sigm',
                  tf.tanh: 'tanh'}
        return to_str[self.nl]


Sigmoid = NL(tf.nn.sigmoid)
Relu = NL(tf.nn.relu)
Tanh = NL(tf.tanh)


class Const():
    def __init__(self, c, trainable=True):
        self.params = [c] if trainable else []
        self.c = c if trainable else tf.stop_gradient(c)
        self.cache = {}

    def __call__(self, n, **kwargs):
        if n not in self.cache:
            c_rep = tf.zeros(tf.stack([n, 1])) + self.c
            self.cache[n] = Struct(y=c_rep)
        return self.cache[n]

#simple matrix multiplication operation (pre-nonlinearity)
class Affine():
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.params = [W, b]
        self.cache = {}

    def __call__(self, x, **kwargs):
        if x not in self.cache:
            xW = tf.matmul(x, self.W)
            y = xW if self.b is None else xW + self.b
            self.cache[x] = Struct(xW=xW, y=y)
        return self.cache[x]

    def to_short_str(self):
        return 'lin{}'.format(self.W.get_shape().as_list()[1])

    @staticmethod
    def build_affine(n1, n2):
        return Affine(W=Glorot(shape=(n1, n2), scale=1.), b=ZeroBias(n2))

#creating the feedforward neural network chain
class Chain():
    def __init__(self, *layers):
        self.layers = layers
        params = sum([layer.params for layer in self.layers], [])
        self.params = list(OrderedDict.fromkeys(params))
        self.cache = {}

    def __call__(self, x, deterministic=False, **kwrags):
        if (x, deterministic) not in self.cache:
            intermediate_outs = OrderedDict()
            y = x
            for layer in self.layers:
                out = layer(y, deterministic=deterministic, **kwrags)
                y = out.y
                intermediate_outs[layer] = out
            self.cache[x] = Struct(intermediate_outs=intermediate_outs, y=y, out=out)

        return self.cache[x]

    @staticmethod
    def build_chain(ns, nonlinearity, last_nonlinearity=None, last_b=None):
        def layer_maker(n1, n2):
            l = [Affine(W=Glorot(shape=(n1, n2), scale=1.), b=ZeroBias(n2))]

            if nonlinearity is not None:
                l.append(nonlinearity)

            return l
        layers = sum([layer_maker(n_in, n_out) for n_in, n_out in zip(ns[:-1], ns[1:-1])], [])
        if last_b is None:
            b = ZeroBias(ns[-1])
        else:
            b = tf.Variable(last_b.astype(np.float32))
        layers.append(Affine(W=Glorot(shape=(ns[-2], ns[-1]), scale=1.), b=b))
        if last_nonlinearity is not None:
            layers.append(nonlinearity)
        return Chain(*layers)


#xl_i ∼ Bernoulli[ψ(hl_i)]
class BernoulliLogits():
    class BernoulliLogitsLogProb():
        def __init__(self, logits):
            self.logits = logits
            self.cache = {}

        def __call__(self, x):
            if x not in self.cache:
                log_prob = -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=x), 1)
                self.cache[x] = log_prob
            return self.cache[x]

    def __init__(self):
        self.params = []
        self.cache = {}

    def __call__(self, logits, **kwargs):
        if logits not in self.cache:
            mu = tf.sigmoid(logits)
            noise = tf.random_uniform(tf.shape(logits))
            sample = tf.cast(tf.less_equal(noise, mu), np.float32)
            log_prob = BernoulliLogits.BernoulliLogitsLogProb(logits)
            self.cache[logits] = Struct(logits=logits, mu=mu, noise=noise, y=sample, sample=sample, log_prob=log_prob)
        return self.cache[logits]
