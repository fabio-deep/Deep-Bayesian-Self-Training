'''
Author: Fabio De Sousa Ribeiro
E-mail: fdesousaribeiro@lincoln.ac.uk
Paper: Deep Bayesian Self-Training
arXiv URL: https://arxiv.org/pdf/1812.01681.pdf
journal URL: https://link.springer.com/article/10.1007/s00521-019-04332-4
'''

import os, shutil, logging
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix

from utils import *
from ops import *

def inverse_uncertainty_weighting(args, idx_to_add, generator, aleatoric, epistemic, alpha=.1, beta=2):
    # calculate predictive uncertainty of acquired unlabelled images
    predictive_uncertainty = (aleatoric[idx_to_add] + epistemic[idx_to_add])**.5 # std

    '''OPTION 1: '''
    # tanh_decay_numer = np.exp(alpha*args.retrain_iter-beta) - np.exp(-(alpha*args.retrain_iter-beta))
    # tanh_decay_denom = np.exp(alpha*args.retrain_iter-beta) + np.exp(-(alpha*args.retrain_iter-beta))
    # weights = (1 - np.log(1 + predictive_uncertainty))**-(tanh_decay_numer / tanh_decay_denom)

    '''OPTION 2: '''
    weights = 1./np.exp(predictive_uncertainty)

    # append sample weights of acquired images to existing .txt training set
    with open(os.path.join(args.confmat_dir, 'train_sample_weights.txt'), 'ab') as f:
        imgnames = [f.split('/')[1] for f in np.array(generator.filenames)[idx_to_add]] # split class/image.jpeg
        np.savetxt(f, np.column_stack([imgnames, weights]),
            delimiter=', ', fmt='%s')

def calculate_uncertainty(MC_preds):
    def aleatoric(MC_log_vars):
        '''aleatoric uncertainty per datapoint'''
        # (N,)
        return np.mean(np.exp(MC_log_vars), axis=0).squeeze() # mean uncertainty of all MC sample log variances

    def epistemic(MC_logits):
        '''epistemic uncertainty per datapoint'''
        # (T, N, K)
        y_hat = np.mean(softmax(MC_logits), axis=0) # mean softmax of all MC sample logits
        y_hat = np.clip(y_hat, 1e-11, 1 - 1e-11) # prevent nans
        # (N,)
        return np.array([-np.sum(y_hat[i] * np.log(y_hat[i])) for i in range(y_hat.shape[0])])#, ndmin=2).T

    #n_classes = MC_preds.shape[-1] // 2 #10
    n_classes = MC_preds.shape[-1] -1 #10
    return aleatoric(MC_preds[...,n_classes:]), epistemic(MC_preds[...,:n_classes]) # (-1)n_classes: and :(-1)n_classes

def calculate_uncertainty_1(MC_preds):
    def aleatoric(y_hat):
        '''aleatoric uncertainty per datapoint'''
        # (T, N, K, K)
        pp_aleatoric = np.zeros(y_hat.shape + y_hat.shape[-1:], dtype=y_hat.dtype)

        for t in range(y_hat.shape[0]): # T samples
            for n in range(y_hat.shape[1]): # N datapoints
                # (K, K)
                pp_aleatoric[t][n] = np.diag(y_hat[t][n]) - y_hat[t][n] * y_hat[t][n].T

        # (N,) <- (N, K, K)
        return np.trace(np.mean(pp_aleatoric, axis=0), axis1=-2, axis2=-1)

    def epistemic(y_hat):
        '''epistemic uncertainty per datapoint'''
        # (T, N, K, 1)
        y_hat = np.expand_dims(y_hat, -1)
        # (N, K, 1)
        mu = np.mean(y_hat, axis=0)
        # (T, N, K, K) broadcasting mu_preds 1->T
        pp_epistemic = (y_hat - mu) * np.moveaxis(y_hat - mu, -2, -1)

        # (N,) <- (N, K, K)
        return np.trace(np.mean(pp_epistemic, axis=0), axis1=-2, axis2=-1)

    return aleatoric(MC_preds), epistemic(MC_preds)

def top_n_predict_labels(top_n, aleatoric, epistemic):
    # get indices of the n datapoints with lowest predictive uncertainty
    predictive_uncertainty = aleatoric + epistemic
    return np.argsort(predictive_uncertainty)[:top_n]

def threshold_predict_labels(args, corrects_aleatoric, corrects_epistemic, aleatoric, epistemic):
    # corrects predictive uncertainty concatenated each split
    corrects_uncertainty = np.concatenate((
        corrects_aleatoric['train'] + corrects_epistemic['train'],
        corrects_aleatoric['valid'] + corrects_epistemic['valid'],
        corrects_aleatoric['test'] + corrects_epistemic['test']), axis=0)

    # predictive uncertainty of unlabelled pool set
    predictive_uncertainty = aleatoric + epistemic
    # predictive_uncertainty = minmax_scale(predictive_uncertainty) # normalise uncertainties [0, 1], optional

    if args.thresh_metric == 'max':
        # get max uncertainty of top n least uncertain correct preds
        thresh = np.max(corrects_uncertainty[np.argsort(corrects_uncertainty)[:args.top_n]])
        # idx of unlabelled images below threshold of predictive uncertainty set by other splits
        return np.where(predictive_uncertainty < thresh)[0], thresh

    elif args.thresh_metric == 'median':
        # get median uncertainty of all corrects, play with mean vs median
        thresh = np.median(corrects_uncertainty)
        # idx of unlabelled images below threshold of predictive uncertainty set by other splits
        return np.where(predictive_uncertainty < thresh)[0], thresh

    elif args.thresh_metric == 'zscore':
        # calculate z score, with mu and sigma defined by corrects
        z_scores = (predictive_uncertainty - np.mean(corrects_uncertainty)) / np.std(corrects_uncertainty) # broadcasting
        std = 1 # threshold, number of standard deviations higher uncertainty than corrects mean uncertainty
        return np.where(z_scores < std)[0], std # get indices within corresponding percentile

    elif args.thresh_metric == 'modified_zscore':
        # uses median and median absolute deviation instead of mean and std
        mad = np.median(np.abs(predictive_uncertainty - np.median(corrects_uncertainty)))
        # the constant .6745 is needed as E[mad] = .6745*sigma for large n
        mad_z_scores = 0.6745 * (predictive_uncertainty - np.median(corrects_uncertainty)) / mad
        thresh = 3 # threshold for outlier detection, 3.5 is common
        return np.where(mad_z_scores < thresh)[0], thresh # get indices of non outliers

    elif args.thresh_metric == 'IQR': # interquartile range used in box plots for outlier detection
        quart_1, quart_3 = np.percentile(corrects_uncertainty, [25, 75]) # set by corrects uncertainty
        # Tukey fence
        iqr = quart_3 - quart_1
        lower_bound = quart_1 - (iqr * 1.5)
        upper_bound = quart_3 + (iqr * 1.5)

        fig, ax = plt.subplots()
        ax.boxplot(corrects_uncertainty, showfliers=False)
        ax.set_title('Self-Training')
        ax.set_xlabel('iteration')
        ax.set_ylabel('predictive uncertainty')
        #plt.show()
        fig.savefig(os.path.join(args.confmat_dir, 'boxplot_iter_'+str(args.retrain_iter)+'.pdf'))
        plt.close(fig=fig)
        thresh = upper_bound #np.median(corrects_uncertainty) #lower_bound
        logging.info('\nIQR uncertainty threshold(std): {:.7f}'.format(thresh**.5))
        return np.where(predictive_uncertainty < thresh)[0], thresh # indices under upper bound

def acquisition(args, idx_to_add, generator, y_hat):
    # get predictions of acquired unlabelled images (indices) only
    y_hat = y_hat[idx_to_add]
    y_true = generator.labels[idx_to_add]
    # file names from unlabelled set to be added to the training set
    files_to_add = np.array(generator.filenames)[idx_to_add]

    # create class directories in annotations dir
    for c in generator.class_indices.keys():
        os.makedirs(os.path.join(args.annotations_dir, c), exist_ok=True)

    for i, img_file in enumerate(files_to_add):
        _, img_name = os.path.split(img_file)
        # turn class predictions to class folder names i.e. 3 -> '3'
        pred_class = str(y_hat[i])

        # copy annotated images from unlabelled pool to annotations dir
        shutil.copy(os.path.join(args.unlabelled_pool_dir, img_file),
            os.path.join(args.annotations_dir, pred_class, img_name))

        # move annotated images from unlabelled pool to training dir
        shutil.move(os.path.join(args.unlabelled_pool_dir, img_file),
            os.path.join(args.train_dir, pred_class, img_name))

    logging.info('{} images moved to train set.\n'.format(len(idx_to_add)))
    logging.info(classification_report(y_true, y_hat, digits=4, labels=np.unique(y_hat)))

    # Plot and save confusion matrix
    fig = plt.figure(figsize=(8, 8))
    plt.grid(False)
    plot_confusion_matrix(confusion_matrix(y_true, y_hat), classes=np.unique(y_hat),
        title='Iteration '+str(args.retrain_iter))#+' Pseudo Labelled Images')#, normalize=True)

    fig.savefig(os.path.join(args.confmat_dir, 'iter_'+str(args.retrain_iter)+'.pdf'))
    plt.close(fig=fig)

def savetxt_uncertainty(args, idx_corrects, idx_acquisitions, generators, corrects_aleatoric, corrects_epistemic, aleatoric, epistemic):
    # makedir if it doesnt exist
    args.savetxt_dir = os.path.join(args.confmat_dir, 'iter_'+str(args.retrain_iter))
    os.makedirs(args.savetxt_dir, exist_ok=True)

    # save the uncertainty (std) and file names of acquired images from unlabelled set
    np.savetxt(os.path.join(args.savetxt_dir, 'acquisitions_aleatoric_uncertainty_iter_'+str(args.retrain_iter)+'.txt'),
        np.column_stack([np.array(generators['unlabelled'].filenames)[idx_acquisitions],
            aleatoric['unlabelled'][idx_acquisitions]]), delimiter=', ', fmt='%s')

    np.savetxt(os.path.join(args.savetxt_dir, 'acquisitions_epistemic_uncertainty_iter_'+str(args.retrain_iter)+'.txt'),
        np.column_stack([np.array(generators['unlabelled'].filenames)[idx_acquisitions],
            epistemic['unlabelled'][idx_acquisitions]]), delimiter=', ', fmt='%s')

    # save aleatoric and epistemic uncertainties for each dataset split
    for (dataset, uncertainty) in aleatoric.items():
        np.savetxt(os.path.join(args.savetxt_dir, dataset+'_aleatoric_uncertainty_iter_'+str(args.retrain_iter)+'.txt'),
            np.column_stack([generators[dataset].filenames, uncertainty]), delimiter=', ', fmt='%s')

    for (dataset, uncertainty) in epistemic.items():
        np.savetxt(os.path.join(args.savetxt_dir, dataset+'_epistemic_uncertainty_iter_'+str(args.retrain_iter)+'.txt'),
            np.column_stack([generators[dataset].filenames, uncertainty]), delimiter=', ', fmt='%s')

    # corrects predictive uncertainty concatenated each split
    corrects_uncertainty = np.concatenate((
        corrects_aleatoric['train'] + corrects_epistemic['train'],
        corrects_aleatoric['valid'] + corrects_epistemic['valid'],
        corrects_aleatoric['test'] + corrects_epistemic['test']), axis=0)

    corrects_filenames = np.concatenate((
        np.array(generators['train'].filenames)[idx_corrects['train']],
        np.array(generators['valid'].filenames)[idx_corrects['valid']],
        np.array(generators['test'].filenames)[idx_corrects['test']]), axis=0)

    # save txt corrects uncertainty to replicate acquisition boxplot
    np.savetxt(os.path.join(args.savetxt_dir, 'validtest_corrects_uncertainty_iter_'+str(args.retrain_iter)+'.txt'),
        np.column_stack([corrects_filenames, corrects_uncertainty]), delimiter=', ', fmt='%s')
