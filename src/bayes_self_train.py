'''
Author: Fabio De Sousa Ribeiro
E-mail: fdesousaribeiro@lincoln.ac.uk
Paper: Deep Bayesian Self-Training
arXiv URL: https://arxiv.org/pdf/1812.01681.pdf
journal URL: https://link.springer.com/article/10.1007/s00521-019-04332-4
'''

import os, sys, cv2, math, h5py, shutil, time, logging, warnings
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from random import shuffle
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers, regularizers, initializers
from tensorflow.keras.regularizers import l2
from tensorflow.keras import applications as architecture
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from ops import *
from layers import *
from utils import *
from uncertainty import *

class SE_DenseNet:
    '''
    Obersevations: https://archive.org/details/github.com-liuzhuang13-DenseNet_-_2017-07-23_18-42-00
    -Wide-DenseNet-BC (L=40, k=36) uses less memory/time while achieving about the same accuracy as DenseNet-BC (L=100, k=12).
    -Wide-DenseNet-BC (L=40, k=48) uses about the same memory/time as DenseNet-BC (L=100, k=12), and is much more accurate.
    '''
    def __init__(self, img_size=(32,32,3), growth_rate=48, trans_ratio=0.5,
        SE_ratio=16, n_classes=10, weight_decay=1e-4, depth=40, n_dense_blocks=3,
        lengthscale=1e-2, n_train_examples=500, is_MC_dropout=True):

        self.k = growth_rate # 12
        self.theta = trans_ratio # .5
        self.SE_ratio = SE_ratio # 16
        self.n_classes = n_classes # 10
        self.weight_decay = l2(weight_decay) #2e-4
        self.img_height = img_size[0]
        self.img_width = img_size[1]
        self.img_depth = img_size[2]
        # true or None, for using dropout at test time
        self.is_MC_dropout = is_MC_dropout

        'Depth must be 3N + 4'
        assert (depth - 4) % 3 == 0
        layers_per_block = int((depth - 4) / 3) // 2 # 6 for L=40, 16 for L=100
        blocks = [layers_per_block for _ in range(n_dense_blocks)]
        self.l = lengthscale # lengthscale hyperparameter to tune
        self.N = n_train_examples
        self.wd = self.l / self.N
        self.dd = 2. / self.N
        # make the model
        self.model = self.build_arch(blocks)

    def build_arch(self, blocks):

        input = Input(shape=(self.img_height, self.img_width, self.img_depth), name='input')

        x = Conv2D(2*self.k, (3, 3), padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=self.weight_decay,
            use_bias=False, name='stem_conv')(input)

        #x = BatchNormalization(epsilon=1.001e-5, name='stem_bn')(x)
        #x = Activation('relu', name='stem_relu')(x)
        #x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='stem_maxpool')(x)

        for i, n_blocks in enumerate(blocks):

            x = self.DenseBlock(x, n_blocks=n_blocks, name='conv'+str(i))
            #x = self.SqueezeExcitation(x, ratio=self.SE_ratio, name='SE_'+str(i))

            if i != len(blocks)-1: # no transition after last denseblock
                x = self.Transition(x, theta=self.theta, name='trans'+str(i))

        # no BN-RELU if using using squeeze excitation block
        x = BatchNormalization(epsilon=1.001e-5, name='bn')(x)
        x = Activation('relu', name='relu')(x)

        x = GlobalAveragePooling2D(name='final_GAP')(x)

        y_hat = Dense(self.n_classes, name='classifier',
            kernel_initializer='he_normal',
            kernel_regularizer=self.weight_decay,
            #bias_regularizer=self.weight_decay,
            #activation='softmax')(x)
            )(x)

        #------predict input noise------
        log_var = Dense(1, name='log_variance',
            kernel_initializer='he_normal',
            kernel_regularizer=self.weight_decay,
            #bias_regularizer=self.weight_decay,
            #activation='softplus')(x)
            )(x)
        y_hat = concatenate([y_hat, log_var])
        #-------------------------------

        model = Model(inputs=input, outputs=y_hat)

        return model

    def DenseBlock(self, in_tensor, n_blocks, name):

        for i in range(n_blocks):
            block_name = name+'_block'+str(i + 1)

            x = BatchNormalization(epsilon=1.001e-5, name=block_name+'_bn0')(in_tensor)
            x = Activation('relu', name=block_name+'_relu0')(x)

            # x = SpatialConcreteDropout(Conv2D(4*self.k, (1, 1), padding='same',
            #     use_bias=False, name=block_name+'_conv1'),
            #     weight_regularizer=self.wd, dropout_regularizer=self.dd)(x)

            x = Conv2D(4*self.k, (1, 1), padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=self.weight_decay,
                use_bias=False, name=block_name+'_conv1')(x)
            x = Dropout(rate=0.2, name=block_name+'_1_dropout')(x, training=self.is_MC_dropout)

            x = BatchNormalization(epsilon=1.001e-5, name=block_name+'_bn1')(x)
            x = Activation('relu', name=block_name+'_relu1')(x)

            # x = SpatialConcreteDropout(Conv2D(self.k, (3, 3), padding='same',
            #     use_bias=False, name=block_name+'_conv2'),
            #     weight_regularizer=self.wd, dropout_regularizer=self.dd)(x)

            x = Conv2D(self.k, (3, 3), padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=self.weight_decay,
               use_bias=False, name=block_name+'_conv2')(x)
            x = Dropout(rate=0.2, name=block_name+'_dropout2')(x, training=self.is_MC_dropout)

            #x = self.SqueezeExcitation(x, ratio=self.SE_ratio, name='SE_'+block_name)

            in_tensor = Concatenate(axis=-1, name=block_name+'_concat')([in_tensor, x])

        return in_tensor

    def Transition(self, in_tensor, theta, name):
        in_channels = in_tensor.shape[-1].value
        compression = int(in_channels * theta)

        x = BatchNormalization(epsilon=1.001e-5, name=name+'_bn')(in_tensor)
        x = Activation('relu', name=name+'_relu')(x)

        x = Conv2D(compression, (1, 1), padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=self.weight_decay,
            use_bias=False, name=name+'_conv')(x)

        x = AveragePooling2D((2, 2), strides=(2, 2), name=name+'_avgpool')(x)

        return x

    def SqueezeExcitation(self, in_tensor, ratio, name):
        in_channels = in_tensor.shape[-1].value
        reduction = int(np.floor(in_channels / ratio))

        in_tensor = BatchNormalization(epsilon=1.001e-5, name=name+'_bn0')(in_tensor)
        in_tensor = Activation('relu', name=name+'_relu0')(in_tensor)

        x = GlobalAveragePooling2D(name=name+'_GAP')(in_tensor)
        x = Reshape((1, 1, in_channels), name=name+'_reshape')(x)

        x = Dense(reduction, name=name+'_fc1',
            kernel_initializer='he_normal',
            kernel_regularizer=self.weight_decay,
            use_bias=False)(x)
        #x = Dropout(rate=0.2, name=name+'_dropout1')(x, training=True)

        x = BatchNormalization(epsilon=1.001e-5, name=name+'_bn1')(x)
        x = Activation('relu', name=name+'_relu1')(x)

        x = Dense(in_channels, name=name+'_fc2',
            kernel_initializer='he_normal',
            kernel_regularizer=self.weight_decay,
            use_bias=False)(x)
        #x = Dropout(rate=0.2, name=name+'_2_dropout')(x, training=True)

        x = BatchNormalization(epsilon=1.001e-5, name=name+'_bn2')(x)
        x = Activation('sigmoid', name=name+'_sigmoid')(x)

        x = multiply([in_tensor, x], name=name+'_mult')

        return x

def Deep_Bayesian_Self_Training(args):

    unlabelled_datagen = ImageDataGenerator()
    with HiddenPrints():
        unlabelled_generator = unlabelled_datagen.flow_from_directory(
            directory=args.unlabelled_pool_dir,
            target_size=(args.img_size[0], args.img_size[1]),
            batch_size=args.batch_size, shuffle=False,
            class_mode='categorical', color_mode='grayscale')

    unlabelled_pool_size = unlabelled_generator.n # n images to label
    args.no_labelling_counter = 0 # n iters without labelling counter
    args.max_iters_without_labelling = 1 # max n iters without labelling before stopping
    args.retrain_iter = 0 # self-training iteration counter

    # placeholders for initial lr and gr
    args.is_MC_dropout = True # use dropout at test time
    args.init_lr = args.learning_rate
    args.init_gr = args.growth_rate
    start_time = time.time()

    '''BEGIN DEEP BAYESIAN SELF-TRAINING LOOP'''
    while unlabelled_pool_size > 0: # while there are images left to label

        args.learning_rate = args.init_lr # reset lr
        args.retrain_iter += 1
        logging.info('\nSelf-Training iter '+str(args.retrain_iter)+':')
        logging.info('-'*21)

        # load datasets splits: ['train', 'valid', 'test', 'unlabelled']
        generators = load_data(args)

        # configure number of acquisitions (top_n), network gr and batch size
        #args.top_n = np.minimum(generators['train'].n // 4, generators['unlabelled'].n)
        args.growth_rate = np.minimum(args.init_gr + (args.retrain_iter - 1) * 12, 24)
        args.batch_size = 32 #if generators['train'].n > 2000 else 16
        #args.lengthscale = generators['train'].n * args.weight_decay

        for key, value in vars(args).items():
            # don't repeatedly print dir paths, except for summaries_dir
            if str(key).split('_')[-1] == 'dir':
                if str(key) != 'summaries_dir':
                    continue # go to next item
            if str(key) != 'sample_weights': #don't print sample weights
                logging.info('--{0}: {1} '.format(str(key), str(value)))

        # train the model with current datasets
        best_model = train(args, generators['train'], generators['valid'], generators['test'])

        # return monte carlo dropout predictions for each dataset split
        MC_preds, generators = MC_dropout(args, model=best_model)

        # evaluate performance on all dataset splits, d=dataset
        acc, y_hat, preds, idx_corrects = {}, {}, {}, {}
        for (d, generator), (_, MC_pred) in zip(generators.items(), MC_preds.items()):
            acc[d], y_hat[d], preds[d], idx_corrects[d] = evaluate(args, generator.labels, MC_pred)

        # calculate predictive uncertainties of all dataset splits, d=dataset
        aleatoric, epistemic = {}, {}
        for (d, MC_pred) in MC_preds.items():
            aleatoric[d], epistemic[d] = calculate_uncertainty(MC_pred)
            #aleatoric[d], epistemic[d] = calculate_uncertainty_1(MC_pred)

        # calculate uncertainties of correct predictions only (idx = idx of corrects)
        corrects_aleatoric, corrects_epistemic = {}, {}
        for (d, MC_pred) in MC_preds.items():
            if d != 'unlabelled': # ['train', 'valid', 'test'], no need for unlabelled split
                corrects_aleatoric[d], corrects_epistemic[d] = calculate_uncertainty(MC_pred[:,idx_corrects[d],:])
                #corrects_aleatoric[d], corrects_epistemic[d] = calculate_uncertainty_1(MC_pred[:,idx_corrects[d],:])

        logging.info('\nTrain | acc: {:.5f} - aleatoric(std): {:.7f} - epistemic(std): {:.7f}'.format(
            acc['train'], aleatoric['train'].mean()**.5, epistemic['train'].mean()**.5))
        logging.info('Valid | acc: {:.5f} - aleatoric(std): {:.7f} - epistemic(std): {:.7f}'.format(
            acc['valid'], aleatoric['valid'].mean()**.5, epistemic['valid'].mean()**.5))
        logging.info('Test | acc: {:.5f} - aleatoric(std): {:.7f} - epistemic(std): {:.7f}'.format(
            acc['test'], aleatoric['test'].mean()**.5, epistemic['test'].mean()**.5))

        '''OPTION 1:''' # indices of 100 datapoints with lowest predictive uncertainty
        # indices = top_n_predict_labels(top_n=args.top_n,
        #     aleatoric=aleatoric['unlabelled'], epistemic=epistemic['unlabelled'])

        '''OPTION 2:''' # indices of datapoints below predictive uncertainty threshold
        indices, thresh = threshold_predict_labels(args=args,
            corrects_aleatoric=corrects_aleatoric, corrects_epistemic=corrects_epistemic,
            aleatoric=aleatoric['unlabelled'], epistemic=epistemic['unlabelled'])

        # move predicted label images to training set
        acquisition(args=args, idx_to_add=indices,
            generator=generators['unlabelled'], y_hat=y_hat['unlabelled']) # argmax pseudo-label predictions

        # evaluate predicted labels vs actual
        cohen_kappa = evaluate_acquisitions(args=args)

        # compute sample weight of acquired images for next self-training iter
        inverse_uncertainty_weighting(args=args, idx_to_add=indices,
            generator=generators['unlabelled'],
            aleatoric=aleatoric['unlabelled'], epistemic=epistemic['unlabelled'])

        # save the uncertainties in txt files for each dataset
        savetxt_uncertainty(args, idx_corrects, indices, generators,
            corrects_aleatoric, corrects_epistemic, aleatoric, epistemic)

        # count how many images are left in the unlabelled pool
        with HiddenPrints():
            unlabelled_datagen = ImageDataGenerator()
            unlabelled_generator = unlabelled_datagen.flow_from_directory(
                directory=args.unlabelled_pool_dir,
                target_size=(args.img_size[0], args.img_size[1]),
                batch_size=args.batch_size, shuffle=False,
                class_mode='categorical', color_mode='grayscale')

        # update number of images left in the pool
        unlabelled_pool_size = unlabelled_generator.n
        logging.info('\n{} unlabelled images left after {} iter.'.format(
            unlabelled_pool_size, args.retrain_iter))

        # stop conditions
        if len(indices) < args.batch_size: # if not enough acquisitions
            args.no_labelling_counter+=1 # increment number of times without acquisitions
            logging.info('\nStop counter {}/{} - not enough unlabelled images found below threshold {:.7f}'.format(
                args.no_labelling_counter, args.max_iters_without_labelling, thresh))
            if args.no_labelling_counter == args.max_iters_without_labelling:
                break

        K.clear_session() # reset keras session

    elapsed = time.time() - start_time
    logging.info('\nTraining hours {}, with {} unlabelled images left.'.format(
        time.strftime("%H:%M:%S", time.gmtime(elapsed)), unlabelled_generator.n))

    return cohen_kappa

def evaluate(args, y_true, MC_preds):
    # (N, K) <- (T, N, K) <- (T, N, K+1)
    preds = np.mean(softmax(MC_preds[...,:args.n_classes]), axis=0)
    y_hat = np.argmax(preds, 1)
    acc = (y_true == y_hat).mean()
    idx_corrects = np.where(y_true == y_hat)[0]
    return acc, y_hat, preds, idx_corrects

def MC_dropout(args, model):
    ''' ---------------------- MONTE CARLO DROPOUT --------------------------'''
     # number of samples to run in parallel
    assert(args.MC_n_samples % args.MC_n_parallel == 0)
    T = args.MC_n_samples // args.MC_n_parallel # 50/10=5

    input = Input(shape=(model.input_shape[1:]))
    x = RepeatImage(args.MC_n_parallel)(input) # monte carlo samples
    x = TimeDistributed(model)(x) # apply model to each element in axis=1 of x
    model_repeated = Model(inputs=input, outputs=x)
    #model_repeated.summary()

    MC_preds, generators, datagens = {}, {}, {}

    logging.info('\nMC Dropout {}/{} samples in parallel:'.format(
        args.MC_n_parallel, args.MC_n_samples))
    logging.info('-'*37)

    # loop through each dataset split
    for dataset in ['train', 'valid', 'test', 'unlabelled']:
        logging.info('{}:'.format(dataset))
        for t in tqdm(range(T)): # parallel run
            with HiddenPrints():
                # reload datasets without shuffling, make indices line-up
                datagens[dataset] = ImageDataGenerator()
                generators[dataset] = datagens[dataset].flow_from_directory(
                    directory=args.datasets_dir[dataset],
                    target_size=(args.img_size[0], args.img_size[1]),
                    batch_size=args.batch_size, shuffle=False, # no shuffle train
                    class_mode='categorical', color_mode='grayscale')

            if t == 0:
                MC_preds[dataset] = model_repeated.predict_generator(
                    normalise(generators[dataset]), steps=len(generators[dataset]))
            else:
                MC_preds[dataset] = np.concatenate((MC_preds[dataset],
                    model_repeated.predict_generator(
                        normalise(generators[dataset]), steps=len(generators[dataset]))), axis=1)
        # (T MC samples, N data points, K classes or K+1 if predicting log var)
        MC_preds[dataset] = np.moveaxis(MC_preds[dataset], 1, 0)

    return MC_preds, generators

def Deep_Ensemble_Self_Training(args):

    unlabelled_datagen = ImageDataGenerator()
    with HiddenPrints():
        unlabelled_generator = unlabelled_datagen.flow_from_directory(
            directory=args.unlabelled_pool_dir,
            target_size=(args.img_size[0], args.img_size[1]),
            batch_size=args.batch_size, shuffle=False,
            class_mode='categorical', color_mode='grayscale')

    unlabelled_pool_size = unlabelled_generator.n # n images to label
    args.no_labelling_counter = 0 # n iters without labelling counter
    args.max_iters_without_labelling = 1 # max n iters without labelling before stopping
    args.retrain_iter = 0 # self-training iteration counter
    args.ensemble_size = 5

    # placeholders for initial lr and gr
    args.is_MC_dropout = None # no dropout at test time
    args.init_lr = args.learning_rate
    args.init_gr = args.growth_rate
    start_time = time.time()

    '''BEGIN DEEP ENSEMBLE SELF-TRAINING LOOP'''
    while unlabelled_pool_size > 0: # while there are images left to label

        args.retrain_iter += 1
        args.growth_rate = np.minimum(args.init_gr + (args.retrain_iter - 1) * 12, 24)
        logging.info('\nSelf-Training iter '+str(args.retrain_iter)+':')
        logging.info('-'*21)

        # train the ensemble and save predictions
        ensemble_preds = {}
        for m in range(args.ensemble_size):
            logging.info('\nTraining Ensemble model {}/{}'.format(m+1, args.ensemble_size))
            args.learning_rate = args.init_lr # reset lr

            # load datasets splits: ['train', 'valid', 'test', 'unlabelled']
            generators = load_data(args)

            for key, value in vars(args).items():
                # don't repeatedly print dir paths, except for summaries_dir
                if str(key).split('_')[-1] == 'dir':
                    if str(key) != 'summaries_dir':
                        continue # go to next item
                if str(key) != 'sample_weights': #don't print sample weights
                    logging.info('--{0}: {1} '.format(str(key), str(value)))

            # train the model with current datasets
            best_model = train(args, generators['train'], generators['valid'], generators['test'])

            ensemble_preds, generators = predict_ensemble(args=args, model=best_model,
                ensemble_preds=ensemble_preds, ensemble_num=m)

            K.clear_session() # reset keras session

        # configure number of acquisitions (top_n), network gr and batch size
        #args.top_n = np.minimum(generators['train'].n // 4, generators['unlabelled'].n)
        args.batch_size = 32 #if generators['train'].n > 2000 else 16

        # evaluate performance on all dataset splits, d=dataset
        acc, y_hat, preds, idx_corrects = {}, {}, {}, {}
        for (d, generator), (_, ensemble_pred) in zip(generators.items(), ensemble_preds.items()):
            acc[d], y_hat[d], preds[d], idx_corrects[d] = evaluate(args, generator.labels, ensemble_pred)

        # calculate predictive uncertainties of all dataset splits, d=dataset
        aleatoric, epistemic = {}, {}
        for (d, ensemble_pred) in ensemble_preds.items():
            aleatoric[d], epistemic[d] = calculate_uncertainty(ensemble_pred)
            #aleatoric[d], epistemic[d] = calculate_uncertainty_1(ensemble_pred)

        # calculate uncertainties of correct predictions only (idx = idx of corrects)
        corrects_aleatoric, corrects_epistemic = {}, {}
        for (d, ensemble_pred) in ensemble_preds.items():
            if d != 'unlabelled': # ['train', 'valid', 'test'], no need for unlabelled split
                corrects_aleatoric[d], corrects_epistemic[d] = calculate_uncertainty(ensemble_pred[:,idx_corrects[d],:])
                #corrects_aleatoric[d], corrects_epistemic[d] = calculate_uncertainty_1(ensemble_pred[:,idx_corrects[d],:])

        logging.info('\nTrain | acc: {:.5f} - aleatoric(std): {:.7f} - epistemic(std): {:.7f}'.format(
            acc['train'], aleatoric['train'].mean()**.5, epistemic['train'].mean()**.5))
        logging.info('Valid | acc: {:.5f} - aleatoric(std): {:.7f} - epistemic(std): {:.7f}'.format(
            acc['valid'], aleatoric['valid'].mean()**.5, epistemic['valid'].mean()**.5))
        logging.info('Test | acc: {:.5f} - aleatoric(std): {:.7f} - epistemic(std): {:.7f}'.format(
            acc['test'], aleatoric['test'].mean()**.5, epistemic['test'].mean()**.5))

        '''OPTION 1:''' # indices of 100 datapoints with lowest predictive uncertainty
        # indices = top_n_predict_labels(top_n=args.top_n,
        #     aleatoric=aleatoric['unlabelled'], epistemic=epistemic['unlabelled'])

        '''OPTION 2:''' # indices of datapoints below predictive uncertainty threshold
        indices, thresh = threshold_predict_labels(args=args,
            corrects_aleatoric=corrects_aleatoric, corrects_epistemic=corrects_epistemic,
            aleatoric=aleatoric['unlabelled'], epistemic=epistemic['unlabelled'])

        # move predicted label images to training set
        acquisition(args=args, idx_to_add=indices,
            generator=generators['unlabelled'], y_hat=y_hat['unlabelled']) # argmax pseudo-label predictions

        # evaluate predicted labels vs actual
        cohen_kappa = evaluate_acquisitions(args=args)

        # compute sample weight of acquired images for next self-training iter
        inverse_uncertainty_weighting(args=args, idx_to_add=indices,
            generator=generators['unlabelled'],
            aleatoric=aleatoric['unlabelled'], epistemic=epistemic['unlabelled'])

        # save the uncertainties in txt files for each dataset
        savetxt_uncertainty(args, idx_corrects, indices, generators,
            corrects_aleatoric, corrects_epistemic, aleatoric, epistemic)

        # count how many images are left in the unlabelled pool
        with HiddenPrints():
            unlabelled_datagen = ImageDataGenerator()
            unlabelled_generator = unlabelled_datagen.flow_from_directory(
                directory=args.unlabelled_pool_dir,
                target_size=(args.img_size[0], args.img_size[1]),
                batch_size=args.batch_size, shuffle=False,
                class_mode='categorical', color_mode='grayscale')

        # update number of images left in the pool
        unlabelled_pool_size = unlabelled_generator.n
        logging.info('\n{} unlabelled images left after {} iter.'.format(
            unlabelled_pool_size, args.retrain_iter))

        # stop conditions
        if len(indices) < args.batch_size: # if not enough acquisitions
            args.no_labelling_counter+=1 # increment number of times without acquisitions
            logging.info('\nStop counter {}/{} - not enough unlabelled images found below threshold {:.7f}'.format(
                args.no_labelling_counter, args.max_iters_without_labelling, thresh))
            if args.no_labelling_counter == args.max_iters_without_labelling:
                break

        K.clear_session() # reset keras session

    elapsed = time.time() - start_time
    logging.info('\nFinished in {:02d}:{:02d}:{:02d}, with {} unlabelled images left.'.format(
        elapsed // 3600, (elapsed % 3600 // 60), elapsed % 60, unlabelled_generator.n))

    return cohen_kappa

def predict_ensemble(args, model, ensemble_preds, ensemble_num):

    logging.info('\nPredicting Ensemble model {}/{}:'.format(ensemble_num+1, args.ensemble_size))
    logging.info('-'*30)

    generators, datagens = {}, {}

    for dataset in ['train', 'valid', 'test', 'unlabelled']:
        with HiddenPrints():
            # reload datasets without shuffling, make indices line-up
            datagens[dataset] = ImageDataGenerator()
            generators[dataset] = datagens[dataset].flow_from_directory(
                directory=args.datasets_dir[dataset],
                target_size=(args.img_size[0], args.img_size[1]),
                batch_size=args.batch_size, shuffle=False, # no shuffle train
                class_mode='categorical', color_mode='grayscale')

        if ensemble_num == 0: # if first ensemble model
            ensemble_preds[dataset] = np.expand_dims(model.predict_generator(
                normalise(generators[dataset]), steps=len(generators[dataset])), axis=0)
        else:
            ensemble_preds[dataset] = np.concatenate((ensemble_preds[dataset],
                np.expand_dims(model.predict_generator(
                    normalise(generators[dataset]), steps=len(generators[dataset])), axis=0)), axis=0)

        logging.info('{}... done.'.format(dataset))
    # (M ensemble size, N data points, K classes or K+1 if predicting log var)
    return ensemble_preds, generators

def Deep_Self_Training(args):

    unlabelled_datagen = ImageDataGenerator()
    with HiddenPrints():
        unlabelled_generator = unlabelled_datagen.flow_from_directory(
            directory=args.unlabelled_pool_dir,
            target_size=(args.img_size[0], args.img_size[1]),
            batch_size=args.batch_size, shuffle=False,
            class_mode='categorical', color_mode='grayscale')

    unlabelled_pool_size = unlabelled_generator.n # n images to label
    args.no_labelling_counter = 0 # n iters without labelling counter
    args.max_iters_without_labelling = 1 # max n iters without labelling before stopping
    args.retrain_iter = 0 # self-training iteration counter

    # placeholders for initial lr and gr
    args.is_MC_dropout = None # no dropout at test time
    args.init_lr = args.learning_rate
    args.init_gr = args.growth_rate
    start_time = time.time()

    '''BEGIN DEEP ENSEMBLE SELF-TRAINING LOOP'''
    while unlabelled_pool_size > 0: # while there are images left to label

        args.learning_rate = args.init_lr # reset lr
        args.retrain_iter += 1
        logging.info('\nSelf-Training iter '+str(args.retrain_iter)+':')
        logging.info('-'*21)

        # load datasets splits: ['train', 'valid', 'test', 'unlabelled']
        generators = load_data(args)

        # configure number of acquisitions (top_n), network gr and batch size
        #args.top_n = np.minimum(generators['train'].n // 4, generators['unlabelled'].n)
        args.growth_rate = np.minimum(args.init_gr + (args.retrain_iter - 1) * 12, 24)
        args.batch_size = 32 #if generators['train'].n > 2000 else 16
        #args.lengthscale = generators['train'].n * args.weight_decay

        for key, value in vars(args).items():
            # don't repeatedly print dir paths, except for summaries_dir
            if str(key).split('_')[-1] == 'dir':
                if str(key) != 'summaries_dir':
                    continue # go to next item
            if str(key) != 'sample_weights': #don't print sample weights
                logging.info('--{0}: {1} '.format(str(key), str(value)))

        # train the model with current datasets
        best_model = train(args, generators['train'], generators['valid'], generators['test'])

        # (N, K+1) predict labels for the unlabelled dataset and add most confident to train set
        preds = best_model.predict_generator(
            normalise(generators['unlabelled']), steps=len(generators['unlabelled']))

        # (N,) <- (N, K) <- (N, K+1)
        preds = softmax(preds[...,:args.n_classes]) # activate
        y_hat = np.argmax(preds, 1) # argmax pseudo-label predictions
        max_p_pred = np.amax(preds, 1) # probability/confidence of pseudo-label predictions

        # get sample indices of high confidence pseudo-label predictions
        indices = np.where(max_p_pred > args.p_thresh)[0] # predicted probability threshold

        # move predicted label images to training set
        acquisition(args=args, idx_to_add=indices,
            generator=generators['unlabelled'], y_hat=y_hat) # argmax pseudo-label predictions

        # evaluate predicted labels vs actual
        cohen_kappa = evaluate_acquisitions(args=args)

        # count how many images are left in the unlabelled pool
        with HiddenPrints():
            unlabelled_datagen = ImageDataGenerator()
            unlabelled_generator = unlabelled_datagen.flow_from_directory(
                directory=args.unlabelled_pool_dir,
                target_size=(args.img_size[0], args.img_size[1]),
                batch_size=args.batch_size, shuffle=False,
                class_mode='categorical', color_mode='grayscale')

        # update number of images left in the pool
        unlabelled_pool_size = unlabelled_generator.n
        logging.info('\n{} unlabelled images left after {} iter.'.format(
            unlabelled_pool_size, args.retrain_iter))

        # stop conditions
        if len(indices) < args.batch_size: # if not enough acquisitions
            args.no_labelling_counter+=1 # increment number of times without acquisitions
            logging.info('\nStop counter {}/{} - not enough unlabelled images found below threshold {:.7f}'.format(
                args.no_labelling_counter, args.max_iters_without_labelling, args.p_thresh)) # predicted probability threshold
            if args.no_labelling_counter == args.max_iters_without_labelling:
                break

        K.clear_session() # reset keras session

    elapsed = time.time() - start_time
    logging.info('\nTraining hours {}, with {} unlabelled images left.'.format(
        time.strftime("%H:%M:%S", time.gmtime(elapsed)), unlabelled_generator.n))

    return cohen_kappa

def load_data(args):
    ''' --------------------------- LOAD DATA -------------------------------'''
    train_datagen = ImageDataGenerator()
    #brightness_range=[.9, 1.1],
    #zoom_range=0.05,
    #rotation_range=90)
    valid_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()
    unlabelled_datagen = ImageDataGenerator()

    with HiddenPrints():
        # # this generator also returns the filenames of each image in batches
        train_generator = train_datagen.flow_from_directory(
            directory=args.train_dir,
            target_size=(args.img_size[0], args.img_size[1]),
            batch_size=args.batch_size, shuffle=True, # remember to keep true
            class_mode='categorical', color_mode='grayscale')

        #this generator also returns the filenames of each image in batches
        # train_generator = GeneratorFilenames(
        # directory=args.train_dir,
        # image_data_generator=train_datagen,
        # target_size=(args.img_size[0], args.img_size[1]),
        # batch_size=args.batch_size, shuffle=True, # remember to keep true
        # class_mode='categorical', color_mode='grayscale')

        valid_generator = valid_datagen.flow_from_directory(
        directory=args.valid_dir,
        target_size=(args.img_size[0], args.img_size[1]),
        batch_size=args.batch_size, shuffle=False,
        class_mode='categorical', color_mode='grayscale')

        test_generator = test_datagen.flow_from_directory(
        directory=args.test_dir,
        target_size=(args.img_size[0], args.img_size[1]),
        batch_size=args.batch_size, shuffle=False,
        class_mode='categorical', color_mode='grayscale')

        unlabelled_generator = unlabelled_datagen.flow_from_directory(
        directory=args.unlabelled_pool_dir,
        target_size=(args.img_size[0], args.img_size[1]),
        batch_size=args.batch_size, shuffle=False,
        class_mode='categorical', color_mode='grayscale')

        logging.info('\nLoading data...')
        logging.info('\ntrain: {} \nvalid: {} \ntest: {} \nunlabelled: {}\n'.format(
        train_generator.n, valid_generator.n, test_generator.n, unlabelled_generator.n))

        if args.retrain_iter == 1: # save initial sample weights as 1s since we know the true label
            imgnames = [f.split('/')[1] for f in train_generator.filenames] # split class/image.jpeg
            np.savetxt(os.path.join(args.confmat_dir, 'train_sample_weights.txt'),
            np.column_stack([imgnames, np.ones(train_generator.n)]), delimiter=', ', fmt='%s')

        # load current sample weights for the training set
        samples, weights = np.loadtxt(os.path.join(args.confmat_dir, 'train_sample_weights.txt'),
        delimiter=', ', dtype='U17, float', usecols=(0,1), unpack=True)
        args.sample_weights = dict(zip(samples, weights))

        # get the class weights scaled by the most represented class
        args.class_weights = get_class_weights(train_generator.labels)

        # get stats to centre the data according to train set
        #x_train_mean, x_train_std = dataset_stats(train_generator)

        generators = {'train': train_generator,
        'valid': valid_generator,
        'test': test_generator,
        'unlabelled': unlabelled_generator}

        return generators

def train(args, train_generator, valid_generator, test_generator):

    ''' ------------------------- TRAIN MODEL ----------------------------'''
    network = SE_DenseNet(img_size=args.img_size, growth_rate=args.growth_rate,
        depth=args.depth, n_dense_blocks=args.n_dense_blocks,
        trans_ratio=args.trans_ratio, SE_ratio=args.SE_ratio,
        n_classes=args.n_classes, weight_decay=args.weight_decay,
        lengthscale=args.lengthscale, n_train_examples=train_generator.n,
        is_MC_dropout=args.is_MC_dropout)

    #if args.retrain_iter == 1:
        #network.model.summary()
    logging.info('\nTotal model params: {:n}'.format(network.model.count_params()))

    #optim = optimizers.Adam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999)
    optim = optimizers.SGD(lr=args.learning_rate, momentum=0.9, nesterov=True)

    network.model.compile(optimizer=optim,
        #loss='categorical_crossentropy', metrics=['accuracy'])
        loss=heteroscedastic_crossentropy, metrics=[acc, aleatoric])

    checkpoint = ModelCheckpoint(
        os.path.join(args.checkpoint_dir, 'iter_'+str(args.retrain_iter)+'checkpoint.h5'),
        monitor='val_acc', verbose=1, save_best_only=True)

    loss_history = LossHistory()

    def checkpoint_decay(epoch):
        # if (epoch == int(np.round(args.n_epochs * .4)) or
        #         epoch == int(np.round(args.n_epochs * .65)) or epoch == int(np.round(args.n_epochs * .9))):
        if epoch == int(np.round(args.n_epochs * .5)) or epoch == int(np.round(args.n_epochs * .75)):
            args.learning_rate /= 10
            K.set_value(network.model.optimizer.lr, args.learning_rate)
            #network.model.lr.set_value(args.learning_rate)
        return K.get_value(network.model.optimizer.lr)

    schedule_lr = LearningRateScheduler(checkpoint_decay, verbose=1)
    #schedule_lr = LearningRateScheduler(exp_decay, verbose=1)
    #schedule_lr = ReduceLROnPlateau(monitor='val_acc',
    #    factor=0.9, patience=15, min_lr=1e-6, verbose=1)

    early_stop = EarlyStopping(monitor='val_acc',
        min_delta=0, patience=50, verbose=1, mode='auto')

    summaries = TrainValTensorBoard(
        log_dir=os.path.join(args.summaries_dir, 'iter_'+str(args.retrain_iter)),
        write_graph=False)

    hist = network.model.fit_generator(normalise(train_generator),#train_normalise(train_generator, args.sample_weights),
        verbose=1, shuffle=True,
        epochs=args.n_epochs,
        class_weight=args.class_weights,
        steps_per_epoch=len(train_generator),
        validation_data=normalise(valid_generator), # (x,y)
        validation_steps=len(valid_generator),
        callbacks=[early_stop, summaries, checkpoint, schedule_lr])

    best_model = load_model(os.path.join(args.checkpoint_dir, 'iter_'+str(args.retrain_iter)+'checkpoint.h5'),
        custom_objects={#'SpatialConcreteDropout': SpatialConcreteDropout},
                        #'acc': acc,
                        #'penalised_binary_crossentropy': penalised_binary_crossentropy})
                        'heteroscedastic_crossentropy': heteroscedastic_crossentropy,
                        'aleatoric': aleatoric, 'acc': acc})

    # dropout_ps = np.asarray([K.eval(layer.p) for layer in best_model.layers if hasattr(layer, 'p')])
    # logging.info('\nLearned Dropout ps:\n{}'.format(dropout_ps))

    score = best_model.evaluate_generator(normalise(test_generator), steps=len(test_generator))
    logging.info('\nTest | loss: {:.5f} - acc: {:5f}'.format(score[0], score[1]))

    return best_model

def evaluate_acquisitions(args):
    # load annotated data and unlabelled pool data
    with HiddenPrints():
        unlabelled_datagen = ImageDataGenerator()
        unlabelled = unlabelled_datagen.flow_from_directory(
            directory=args.unlabelled_backup_dir, # directory of backup unlabelled pool
            target_size=(args.img_size[0], args.img_size[1]),
            batch_size=args.batch_size, shuffle=False,
            class_mode='categorical', color_mode='grayscale')

        annotated_datagen = ImageDataGenerator()
        annotated = annotated_datagen.flow_from_directory(
            directory=args.annotations_dir,
            target_size=(args.img_size[0], args.img_size[1]),
            batch_size=args.batch_size, shuffle=False,
            class_mode='categorical', color_mode='grayscale')

    # split unlabelled image names from folder
    u = np.array([os.path.split(unlabelled.filenames[i])[1] for i in range(unlabelled.n)])

    idx = []
    # check annotated images with the labels in unlabelled pool dir
    for i in range(annotated.n):
        # repeat annotated image filename unlabelled.n times, and find it in the pool
        a = np.repeat(os.path.split(annotated.filenames[i])[1], unlabelled.n)
        # find corresponding annotated image in the unlabelled pool
        idx.append(np.array(np.where(u == a)[0]))

    cohen_kappa = cohen_kappa_score(unlabelled.labels[idx], annotated.labels)

    logging.info('\n{}/{} annotated images cohen\'s kappa score: {:.4f}\n'.format(
        annotated.n, unlabelled.n, cohen_kappa))
    logging.info(classification_report(unlabelled.labels[idx], annotated.labels,
        digits=4, labels=np.unique(annotated.labels)))

    # backup annotation dir at each acquisition iteration
    shutil.copytree(args.annotations_dir,
        args.annotations_dir+'_iter'+str(args.retrain_iter))

    return cohen_kappa
