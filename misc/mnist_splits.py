import os, cv2
import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

dataset = 'MNIST_standard'
working_dir = os.path.join(os.path.split(os.getcwd())[0], 'data', dataset)

def save_images(x_train, y_train, x_test, y_test):
    '''One-time function for saving the images to disk.'''
    for i in range(x_train.shape[0]):
        img_path = os.path.join(working_dir, 'train', str(y_train[i]), str(y_train[i])+'_img'+str(i)+'.jpeg')
        cv2.imwrite(img_path, x_train[i], [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    for i in range(x_test.shape[0]):
        img_path = os.path.join(working_dir,'test', str(y_test[i]), str(y_test[i])+'_img'+str(i)+'.jpeg')
        cv2.imwrite(img_path, x_test[i], [int(cv2.IMWRITE_JPEG_QUALITY), 100])

save_images(x_train, y_train, x_test, y_test)

dataset = 'MNIST_unlabelled'
working_dir = os.path.join(os.path.split(os.getcwd())[0], 'data', dataset)

def split_unlabelled_pool(x_train, y_train, x_test, y_test):
    '''Splitting the training dataset into a small initial training set
    (train_init), validation set (valid) and unlabelled pool (unlabelled)'''
    n_classes = 10
    n_train_init = 500 // n_classes # number of examples in initial training set
    n_valid = 5000 // n_classes # number of examples in validation set
    # the unlabelled pool consists of all remaining examples

    for class_i in range(n_classes):
        # get training example indices pertaining to each class
        idx = np.argwhere(y_train==class_i).squeeze()
        np.random.shuffle(idx)

        train_init = idx[:n_train_init] # allocate initial training set
        valid = idx[n_train_init:n_valid+n_train_init] # allocate validation set
        unlabelled = idx[n_valid+n_train_init:] # allocate unlabelled pool

        # save images into correct folders for each split
        for img_idx in train_init:
            img_path = os.path.join(working_dir,'train_init', str(class_i), str(class_i)+'_img'+str(img_idx)+'.jpeg')
            cv2.imwrite(img_path, x_train[img_idx], [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        for img_idx in valid:
            img_path = os.path.join(working_dir,'valid', str(class_i), str(class_i)+'_img'+str(img_idx)+'.jpeg')
            cv2.imwrite(img_path, x_train[img_idx], [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        for img_idx in unlabelled:
            img_path = os.path.join(working_dir,'unlabelled', str(class_i), str(class_i)+'_img'+str(img_idx)+'.jpeg')
            cv2.imwrite(img_path, x_train[img_idx], [int(cv2.IMWRITE_JPEG_QUALITY), 100])

split_unlabelled_pool(x_train, y_train, x_test, y_test)
