'''
Author: Fabio De Sousa Ribeiro
E-mail: fdesousaribeiro@lincoln.ac.uk
Paper: Deep Bayesian Self-Training
arXiv URL: https://arxiv.org/pdf/1812.01681.pdf
journal URL: https://link.springer.com/article/10.1007/s00521-019-04332-4
'''

import math, operator
import numpy as np
import tensorflow as tf
from collections import Counter
import tensorflow.keras.backend as K
from tensorflow.contrib import distributions

def heteroscedastic_crossentropy(y_true, logits_log_var):
    def monte_carlo(T, logits, gaussian):
        T_softmax = K.zeros_like(logits)
        n_classes = logits.shape[-1]

        for i in range(T):
            # (?, K) <- (K, ?) <- (K, ?, 1)
            noise = K.transpose(K.squeeze(gaussian.sample(n_classes), axis=-1)) # draw a sample per logit
            #noise = gaussian.sample() # draw sample from multivariate, for all logits at once
            T_softmax += K.softmax(logits + noise)
        # (?, K)
        return (1/T) * T_softmax

    n_classes = logits_log_var.shape[-1] -1 #10
    #n_classes = logits_log_var.shape[-1] // 2 #10
    std = K.sqrt(K.exp(logits_log_var[:,n_classes:]))

    # get T softmax monte carlo simulations
    y_hat = monte_carlo(T=100, # number of simulations
        logits=logits_log_var[:,:n_classes], # logits
        gaussian=tf.distributions.Normal(loc=K.zeros_like(std), scale=std)) # log_var to std

    y_hat = K.clip(y_hat, 1e-11, 1 - 1e-11) # prevent nans
    #beta = 1.
    #gamma = .1
    #H = -K.sum(y_hat * K.log(y_hat), -1) # entropy term to punish confident predictions
    nll = -K.sum(y_true * K.log(y_hat), -1) # negative log likelihood
    return nll
    #return nll - beta * H
    #return K.mean(ll - beta * K.max(0, gamma - H))

def penalised_categorical_crossentropy(y_true, y_hat):
    beta = .1
    #gamma = .1
    y_hat = K.clip(y_hat, 1e-11, 1 - 1e-11)
    H = -K.sum(y_hat * K.log(y_hat), 1) # entropy term to punish confident predictions
    nll = -K.sum(y_true * K.log(y_hat), 1) # negative log likelihood
    return K.mean(nll - beta * H)
    #return K.mean(ll - beta * K.max(0, gamma - H))

def penalised_binary_crossentropy(y_true, y_hat):
    beta = .1 # weight hyperparameter
    y_hat = K.clip(y_hat, 1e-11, 1. - 1e-11)
    # entropy regularisation term, punish high confidence values
    H = -y_hat * K.log(y_hat) - (1. - y_hat) * K.log(1. - y_hat)
    nll = -y_true * K.log(y_hat) - (1. - y_true) * K.log(1. - y_hat) # negative log likelihood
    return K.mean(nll - beta * H)

def acc(y_true, logits_log_var):
    y_hat = K.softmax(logits_log_var[...,:-1]) # 10 classes
    return K.cast(K.equal(K.argmax(y_true), K.argmax(y_hat)),K.floatx()) # categorical

def aleatoric(y_true, logits_log_var): # metric for printing during training
    return K.exp(logits_log_var[...,-1:]) # n_classes:

def softmax(logits):
    e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    s = np.sum(e, axis=-1, keepdims=True)
    return e / s

def train_normalise(generator, sample_weights): # generator, dict
    # yields normalised samples in batches along with respective sample weights
    for (x_y, filenames) in generator:
        x_mu = np.mean(x_y[0], axis=(1,2,3), keepdims=True)
        x_std = np.std(x_y[0], axis=(1,2,3), keepdims=True)
        # get corresponding saved sample weights given batch filenames as dict keys
        f = operator.itemgetter(*filenames)
        weights = np.array(f(sample_weights))
        # (x, y, w)
        yield (x_y[0] - x_mu) / (x_std + 1e-11), x_y[1], weights

def normalise(generator): # generator
    # yields normalised samples in batches
    for (x, y) in generator:
        x_mu = np.mean(x, axis=(1,2,3), keepdims=True)
        x_std = np.std(x, axis=(1,2,3), keepdims=True)
        yield (x - x_mu) / (x_std + 1e-11), y

def minmax_scale(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x

def limit_mem():
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))

def exp_decay(epoch):
   initial_lr = learning_rate
   k = 1e-3
   lr = initial_lr * np.exp(-k*epoch)
   return lr

def step_decay(epoch):
   initial_lr = learning_rate
   drop = 0.5
   epochs_drop = 10.0
   lr = initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
   return lr

def get_class_weights(class_freq):
  counter = Counter(class_freq)
  majority = max(counter.values())
  return  {cls: float(majority/count) for cls, count in counter.items()}

def dataset_stats(generator):
  running_mean = 0.
  running_std = 0.

  for i in range(len(generator)):
      running_mean += np.mean(generator[i][0], axis=(0,1,2)) * generator[i][0].shape[0]
      running_std += np.std(generator[i][0], axis=(0,1,2)) * generator[i][0].shape[0]

      return running_mean / generator.n, running_std / generator.n
