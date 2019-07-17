'''
Author: Fabio De Sousa Ribeiro
E-mail: fdesousaribeiro@lincoln.ac.uk
Paper: Deep Bayesian Self-Training
arXiv URL: https://arxiv.org/pdf/1812.01681.pdf
journal URL: https://link.springer.com/article/10.1007/s00521-019-04332-4
'''

import os, warnings, logging, argparse
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from bayes_self_train import *

def main(args):
    # keep GPU directories separate for convenience
    GPU_dir = 'GPU_'+str(args.GPU)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.GPU)

    ''''--------------------------- DATASETS --------------------------------'''
    if args.dataset == 'fashionmnist':
        dataset = 'FashionMNIST_unlabelled'
        working_dir = os.path.join(
            os.path.split(os.getcwd())[0], 'data', GPU_dir, dataset)

    elif args.dataset == 'mnist':
        dataset = 'MNIST_unlabelled'
        working_dir = os.path.join(
            os.path.split(os.getcwd())[0], 'data', GPU_dir, dataset)

    ''''----------------------- EXPERIMENT CONFIG ---------------------------'''
    experiment_dir = os.path.join(
        os.path.split(os.getcwd())[0], 'experiments', GPU_dir)
    os.makedirs(experiment_dir, exist_ok=True)

    # check number of models already saved in Experiments dir, add 1 to get new model number
    model_num = len(os.listdir(experiment_dir)) + 1

    # create all save dirs
    args.model_dir = os.path.join(experiment_dir,'Model_'+str(model_num))
    os.makedirs(args.model_dir, exist_ok=True)

    args.annotations_dir = os.path.join(args.model_dir, 'annotations')
    os.makedirs(args.annotations_dir, exist_ok=True)

    args.confmat_dir = os.path.join(args.model_dir, 'confusion_matrices')
    os.makedirs(args.confmat_dir, exist_ok=True)

    args.checkpoint_dir = os.path.join(args.model_dir, 'checkpoints')
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    args.summaries_dir = os.path.join(args.model_dir, 'summaries')
    os.makedirs(args.summaries_dir, exist_ok=True)

    # save hyperparameter arguments
    with open(os.path.join(args.model_dir, 'hyperparams.txt'), 'w') as logs:
        for key, value in vars(args).items():
            logs.write('--{0}={1}'.format(str(key), str(value)))

    # reset root logger
    [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]
    # info logger for saving command line outputs during training
    logging.basicConfig(level=logging.INFO, format='%(message)s',
         handlers=[logging.FileHandler(os.path.join(args.model_dir, 'trainlogs.txt')),
            logging.StreamHandler()])

    for key, value in vars(args).items():
        # don't print dir paths, except for summaries_dir
        if str(key).split('_')[-1] == 'dir':
            if str(key) != 'summaries_dir':
                continue # go to next item
        logging.info('--{0}: {1} '.format(str(key), str(value)))

    # define the various paths needed
    args.train_dir = os.path.join(working_dir, 'train_added')
    args.valid_dir = os.path.join(working_dir, 'valid')
    args.test_dir = os.path.join(working_dir, 'test')
    args.unlabelled_pool_dir = os.path.join(working_dir, 'unlabelled')
    args.unlabelled_backup_dir = os.path.join(os.path.split(
        os.getcwd())[0], 'data', 'backups', dataset, 'unlabelled')

    args.datasets_dir = {'train': args.train_dir,
                        'valid': args.valid_dir,
                        'test': args.test_dir,
                        'unlabelled': args.unlabelled_pool_dir}

    ''' -------------------------- TRAIN MODEL -------------------------------'''
    if args.mode == 'DBST':
        cohen_kappa_score = Deep_Bayesian_Self_Training(args)
    elif args.mode == 'DEST':
        cohen_kappa_score = Deep_Ensemble_Self_Training(args)
    elif args.mode == 'DST':
        cohen_kappa_score = Deep_Self_Training(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU', type=int, default=0)
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--growth_rate', type=int, default=12)
    parser.add_argument('--n_dense_blocks', type=int, default=3)
    parser.add_argument('--depth', type=int, default=40)
    parser.add_argument('--SE_ratio', type=int, default=16)
    parser.add_argument('--MC_n_samples', type=int, default=30)
    parser.add_argument('--top_n', type=int, default=100)
    parser.add_argument('--thresh_metric', default='IQR')
    parser.add_argument('--MC_n_parallel', type=int, default=10)
    parser.add_argument('--trans_ratio', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lengthscale', type=float, default=1e-2)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--img_size', nargs='+', type=int, default=(28,28,1))
    parser.add_argument('--mode', default='DBST')
    parser.add_argument('--p_thresh', type=float, default=0.99)

    #main(parser.parse_args())
    score = main(parser.parse_known_args()[0])
    args = parser.parse_known_args()[0]
    print(score)
