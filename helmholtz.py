import tensorflow as tf
import numpy as np
import datasets
from utils import config, Struct
import utils
import os
import sys
from nnet import Affine, Chain, Sigmoid, BernoulliLogits, Const, ZeroBias
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Model():
    def __init__(self, x, latent_units):
        self.x = x
        self.mb_size = tf.shape(x)[0]
        self.latent_units = latent_units
        datadim = self.x.get_shape().as_list()[1]
        #recognition units
        self.q_units = []
        for n1, n2 in zip([datadim] + latent_units, latent_units):
            self.q_units.append([n1] + [n2])
        #generation units
        self.p_units = []
        for n1, n2 in zip(reversed(latent_units), list(reversed(latent_units))[1:] + [datadim]):
            self.p_units.append([n1] + [n2])
        #setting up the neural network connections (generative and recognition)
        self._setup_connectivity()
        #trains only p_units
        self._wake_phase()
        #trains only q_units
        self._sleep_phase()
        #tensorflow optimizer calculates gradients on this loss
        self.loss = - (self.log_q_sleep + self.log_p_wake)
        self._test_loss()

    def _setup_connectivity(self):
        self.q = []
        for units in self.q_units:
            self.q.append(
                Chain(Chain.build_chain(units, Sigmoid),
                      BernoulliLogits()
                      )
            )
        self.p = []
        for units in self.p_units:
            self.p.append(
                Chain(Chain.build_chain(units, Sigmoid),
                      BernoulliLogits()
                      )
            )
        self.prior = Chain(Const(ZeroBias(self.latent_units[-1])),
                           BernoulliLogits()
                           )

    def _wake_phase(self):
        self.q_samples = [self.x]
        for q in self.q:
            self.q_samples.append(tf.stop_gradient(q(self.q_samples[-1]).out.sample))
        self.log_p_wake = self.prior(self.mb_size).out.log_prob(self.q_samples[-1])
        for p, sample1, sample2 in zip(reversed(self.p), self.q_samples[1:], self.q_samples):
            self.log_p_wake += p(sample1).out.log_prob(sample2)

    def _sleep_phase(self):
        self.sleep_samples = [self.q_samples[-1]]
        for p in self.p:
            self.sleep_samples.append(p(self.sleep_samples[-1]).out.sample)

        self.log_q_sleep = 0
        for q, p, sample in zip(reversed(self.q), self.p, self.sleep_samples):
            sample1 = tf.stop_gradient(p(sample).out.sample)
            self.log_q_sleep += q(sample1).out.log_prob(sample)

    def _test_loss(self):
        self.k = tf.placeholder(tf.int32)
        self.x_rep = utils.tf_repeat(self.x, self.k)
        self.q_samples_rep = [self.x_rep]
        for q in self.q:
            self.q_samples_rep.append(q(self.q_samples_rep[-1]).out.sample)

        #defining ELBO
        self.variational_lower_bound = self.prior(self.k*self.mb_size).out.log_prob(self.q_samples_rep[-1])
        for p, sample1, sample2 in zip(reversed(self.p), self.q_samples_rep[1:], self.q_samples_rep):
            self.variational_lower_bound += p(sample1).out.log_prob(sample2)
        for q, sample1, sample2 in zip(self.q, self.q_samples_rep, self.q_samples_rep[1:]):
            self.variational_lower_bound -= q(sample1).out.log_prob(sample2)

        self.variational_lower_bound = tf.reshape(self.variational_lower_bound, tf.stack([-1, self.k]))
        self.variational_lower_bound = utils.tf_log_mean_exp(self.variational_lower_bound)

        self.reconstruction_loss = self.p[-1](self.q_samples_rep[1]).out.log_prob(self.x_rep)
        self.reconstruction_loss = tf.reshape(self.reconstruction_loss, tf.stack([-1, self.k]))
        self.reconstruction_loss = tf.reduce_mean(self.reconstruction_loss, 1)

#calculates and writes intermediate MNIST validation loss in results directory
def results_writer(sess, model, dataset, directory, epoch):
    x_np, _ = dataset.get_random_minibatch('train', 100, np.random.RandomState(123))
    x_np_valid, _ = dataset.get_random_minibatch('valid', 100, np.random.RandomState(123))

    recs = sess.run(model.p[-1](model.q[0](model.x).out.sample).out.mu, feed_dict={model.x: x_np})
    recs = utils.misc.tile_images(dataset.reshape_for_display(recs))
    orig = utils.misc.tile_images(dataset.reshape_for_display(x_np))
    orig_and_recs = np.concatenate([orig, recs], axis=1)
    plt.imshow(orig_and_recs, cmap='gray')
    plt.savefig(os.path.join(directory, 'orig_and_reconstruction.png'))
    plt.close()

    loss_valid, rec_valid = sess.run([model.variational_lower_bound, model.reconstruction_loss],feed_dict={model.x: x_np_valid, model.k: 100})

    print('epoch ', epoch,' valid rec100', np.mean(rec_valid),' valid loss100', np.mean(loss_valid))
    f1 = open(os.path.join(directory,'reconstruction-loss.txt'), "a")
    f1.write(str(np.mean(rec_valid)) + " ")
    f2 = open(os.path.join(directory,'variational-loss.txt'), "a")
    f2.write(str(np.mean(loss_valid)) + " ")

#function which evaluates network on the MNIST test dataset, k is the number of data samples
def measure_test_log_likelihood(sess, model, dataset, directory, k=5000):
    mbsize = 10
    ll = 0.
    n = 0
    for x_np, _ in dataset.all_minibatches('test', mbsize, np.random.RandomState(123)):
        n += x_np.shape[0]
        ll += np.sum(sess.run(model.variational_lower_bound, feed_dict={model.x: x_np, model.k: k}))
        sys.stdout.write(str(n)+','+str(ll/n))
    utils.print_and_save(os.path.join(directory, 'loss.txt'),
                         'test log likelihood 5000', ll/n)


if __name__ == '__main__':
    # set parameters of the network here:
    default_args = Struct(

        # to set optimizer to sgd, use : tf.train.GradientDescentOptimizer
        optimizer=tf.train.AdamOptimizer,
        #mini-batch size
        mb_size=200,
        #
        latent_units=[100,50,20],
        #batch-normalization
        bn=True,
        #learning rate
        lr=0.001,
        #number of epochs to train
        n_epochs=10)

    #to view final results
    directory = os.path.join(config.RESULTSDIR)
    if not os.path.exists(directory):
        os.makedirs(directory)

    #using binary MNIST dataset
    dataset = datasets.MNIST(binary=True)

    x = tf.placeholder(np.float32, shape=(None, dataset.get_data_dim()))

    #creating a model instance
    model = Model(x, default_args.latent_units)
    examples_per_epoch = dataset.data['train'][0].shape[0]
    num_updates = int(default_args.n_epochs * examples_per_epoch / default_args.mb_size)

    #step variable keeps track of the number of batches seen so far, during training
    step = tf.Variable(0, trainable=False)
    #set learning rate in the default_args Struct above
    lr = tf.placeholder(tf.float32)
    train_op = default_args.optimizer(lr).minimize(model.loss, global_step=step)

    init_op = tf.global_variables_initializer()

    #here is where the actual training takes place!
    with tf.Session() as sess:
        sess.run(init_op)

        for x_np, _ in dataset.random_minibatches('train', default_args.mb_size, num_updates):
            i, _ = sess.run([step, train_op], feed_dict={x: x_np, lr: default_args.lr})
            if i % 250 == 1 or i == num_updates - 1:
                results_writer(sess, model, dataset, directory, float(i) * default_args.mb_size / examples_per_epoch)