import morb
from morb import rbms, stats, updaters, trainers, monitors

import theano
import theano.tensor as T

import numpy as np

import gzip, cPickle

import matplotlib.pyplot as plt
plt.ion()

from test_utils import generate_data, get_context

# DEBUGGING

from theano import ProfileMode
# mode = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())
# mode = theano.compile.DebugMode(check_py_code=False, require_matching_strides=False)
mode = None


# load data
print ">> Loading dataset..."

f = gzip.open('../../data/mnist.pkl.gz','rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_set_x, train_set_y = train_set
valid_set_x, valid_set_y = valid_set
test_set_x, test_set_y = test_set


# TODO DEBUG
# train_set_x = train_set_x[:10000]
valid_set_x = valid_set_x[:1000]


n_visible = train_set_x.shape[1]
n_hidden = 500
mb_size = 1
n_chains = 20
k = 1
learning_rate = 0.001
epochs = 10

np_beta = np.arange(1, 0, -1.0 / n_chains, dtype=theano.config.floatX)[:, np.newaxis]
#np_beta = np.ones(n_chains, dtype=theano.config.floatX)[:, np.newaxis]
# should be broadcastable row
beta = theano.shared.constructors[2](np_beta, broadcastable=(False, True))


print ">> Constructing RBM..."
rbm = rbms.BinaryBinaryRBM(n_visible, n_hidden)
initial_vmap = { rbm.v: T.matrix('v') }

persistent_vmap = { rbm.h: theano.shared(np.zeros((n_chains, n_hidden), dtype=theano.config.floatX)) }

# try to calculate weight updates using CD stats
print ">> Constructing contrastive divergence updaters..."
s = stats.pt_stats(rbm, initial_vmap, visible_units=[rbm.v], hidden_units=[rbm.h], k=k, persistent_vmap=persistent_vmap, beta=beta, m=1)

umap = {}
for var in rbm.variables:
    pu = var + (learning_rate / float(mb_size)) * updaters.CDUpdater(rbm, var, s)
    umap[var] = pu

print ">> Compiling functions..."
t = trainers.MinibatchTrainer(rbm, umap)

m_data = s['data'][rbm.v]
m_model = s['model'][rbm.v]
e_data = rbm.energy(s['data'])
e_model = rbm.energy(s['model'])

train = t.compile_function(initial_vmap, mb_size=mb_size, monitors=[e_data, e_model], name='train', mode=mode)
evaluate = t.compile_function(initial_vmap, mb_size=mb_size, monitors=[m_data, m_model, e_data, e_model], name='evaluate', train=False, mode=mode)


def plot_data(d):
    plt.figure(5)
    plt.clf()
    plt.imshow(d.reshape((28,28)), interpolation='gaussian')
    plt.draw()


# TRAINING 

print ">> Training for %d epochs..." % epochs

edata_train_so_far = []
emodel_train_so_far = []
edata_so_far = []
emodel_so_far = []

for epoch in range(epochs):
    monitoring_data_train = [(energy_data, energy_model) for energy_data, energy_model in train({ rbm.v: train_set_x })]
    edata_train_list, emodel_train_list = zip(*monitoring_data_train)
    edata_train = np.mean(edata_train_list)
    emodel_train = np.mean(emodel_train_list)
    
    monitoring_data = [(data, model, energy_data, energy_model) for data, model, energy_data, energy_model in evaluate({ rbm.v: valid_set_x })]
    vdata, vmodel, edata, emodel = zip(*monitoring_data)
    edata_valid = np.mean(edata)
    emodel_valid = np.mean(emodel)
    
    # plotting
    edata_so_far.append(edata_valid)
    emodel_so_far.append(emodel_valid)
    edata_train_so_far.append(edata_train)
    emodel_train_so_far.append(emodel_train)
    
    
    plt.figure(4)
    plt.clf()
    plt.plot(edata_so_far, label='validation / data')
    plt.plot(emodel_so_far, label='validation / model')
    plt.plot(edata_train_so_far, label='train / data')
    plt.plot(emodel_train_so_far, label='train / model')
    plt.title("energy")
    plt.legend()
    plt.draw()
    
    # plot some samples
    plt.figure(2)
    plt.clf()
    plt.imshow(vdata[0][0].reshape((28, 28)))
    plt.draw()
    plt.figure(3)
    plt.clf()
    plt.imshow(vmodel[0][0].reshape((28, 28)))
    plt.draw()

    
    print "Epoch %d" % epoch
    print "training set: data energy = %.2f, model energy = %.2f" % (edata_train, emodel_train)
    print "validation set: data energy = %.2f, model energy = %.2f" % (edata_valid, emodel_valid)




