import csv
import time
import argparse
import os
import re
from time import strftime
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import color_codes, get_dirs, get_int
from utils import get_mask, get_normalised_image
from utils import time_to_string, find_file
from models import SimpleUNet
from datasets import LesionCroppingDataset, LesionDataset

"""
> Arguments
"""


def parse_inputs():
    # I decided to separate this function, for easier acces to the command line parameters
    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')

    # Mode selector
    parser.add_argument(
        '-d', '--data-directory',
        dest='data_dir', default=[
            '/home/mariano/data/DiceProject/cross-sectional',
            '/home/mariano/data/DiceProject/longitudinal'
        ],
        help='Option to define the folders for each ask with all the patients.'
    )
    parser.add_argument(
        '-e', '--epochs',
        dest='epochs',
        type=int, default=100,
        help='Number of epochs'
    )
    parser.add_argument(
        '-b', '--batch-size',
        dest='batch_size',
        type=int, default=8,
        help='Number of samples per batch'
    )
    parser.add_argument(
        '-k', '--patch-size',
        dest='patch_size',
        type=int, default=32,
        help='Patch size'
    )

    options = vars(parser.parse_args())

    return options


"""
> Data functions
"""


def get_images(d_path, image_tags=None, verbose=0):
    """
    Function to get all the images from a folder. For training, files are
    loaded from the loo_dir, which is the one used for a leave-one-out train
    and for testing val_dir is used.
    :param d_path:
    :param image_tags:
    :param verbose: Verbosity level.
    :return:
    """
    c = color_codes()
    if image_tags is None:
        image_tags = ['t1', 'flair']
    tag_string = '(' + '|'.join(image_tags) + ')'
    patients = sorted(get_dirs(d_path), key=get_int)
    patient_dicts = []
    test_start = time.time()

    n_images = 0

    for pi, p in enumerate(patients):
        p_path = os.path.join(d_path, p)
        tests = len(patients) - pi
        if verbose > 0:
            test_elapsed = time.time() - test_start
            test_eta = tests * test_elapsed / (pi + 1)
            print(
                '{:}Loading patient {:} ({:d}/{:d}) '
                '{:} ETA {:}'.format(
                    c['clr'], p, pi + 1, len(patients),
                    time_to_string(test_elapsed),
                    time_to_string(test_eta),
                ), end='\r'
            )

        find_file('(' + '|'.join(image_tags) + ')', p_path)
        files = [
            os.path.join(p_path, file) for file in os.listdir(p_path)
            if not os.path.isdir(file) and re.search(tag_string, file)
        ]
        brain = get_mask(os.path.join(p_path, 'brain.nii.gz'))
        lesion = get_mask(os.path.join(p_path, 'lesion.nii.gz'))
        images = [
            get_normalised_image(file, brain, dtype=np.float16)
            for file in files
        ]
        n_images = len(images)
        p_dict = {
            'name': p,
            'brain': brain,
            'lesion': lesion,
            'images': np.concatenate(images, axis=0),
        }
        patient_dicts.append(p_dict)

    if verbose > 0:
        print('{:}All patients loaded'.format(c['clr']))

    return patient_dicts, n_images


"""
> Network functions
"""


def train(
        d_path, net, model_name, train_dicts, test_dicts, negative_ratio=1,
        log_file=None, verbose=0
):
    """

    :param d_path:
    :param net:
    :param model_name:
    :param train_dicts:
    :param test_dicts:
    :param negative_ratio:
    :param log_file:
    :param verbose:
    :return:
    """
    # Init
    c = color_codes()
    options = parse_inputs()
    epochs = options['epochs']
    batch_size = options['batch_size']
    patch_size = options['patch_size']
    overlap = patch_size // 2

    try:
        net.load_model(os.path.join(d_path, model_name))
    except IOError:

        if verbose > 1:
            print('Preparing the training datasets / dataloaders')

        # Here we'll do the training / validation split...
        d_train = d_val = [t['images'] for t in train_dicts]
        r_train = r_val = [t['brain'] for t in train_dicts]
        m_train = m_val = [t['lesion'] for t in train_dicts]
        d_test = [t['images'] for t in test_dicts]
        r_test = [t['brain'] for t in test_dicts]
        m_test = [t['lesion'] for t in test_dicts]

        # Training
        if verbose > 1:
            print('< Training dataset >')
        train_dataset = LesionCroppingDataset(
            d_train, m_train, r_train, patch_size=patch_size,
            overlap=overlap, negative_ratio=negative_ratio
        )

        if verbose > 1:
            print('Dataloader creation <with validation>')
        train_loader = DataLoader(train_dataset, batch_size, True)

        # Validation
        if verbose > 1:
            print('< Validation dataset >')
        val_dataset = LesionDataset(d_val, m_val, r_val)
        if verbose > 1:
            print('Dataloader creation <val>')
        val_loader = DataLoader(val_dataset, 1)

        # Test
        if verbose > 1:
            print('< Test dataset >')
        test_dataset = LesionDataset(d_test, m_test, r_test)
        if verbose > 1:
            print('Dataloader creation <val>')
        test_loader = DataLoader(test_dataset, 1)

        if verbose > 1:
            print(
                'Training / validation / test samples = '
                '{:d} / {:d} / {:d}'.format(
                    len(train_dataset), len(val_dataset), len(test_dataset)
                )
            )
        if verbose > 0:
            n_param = sum(
                p.numel() for p in net.parameters()
                if p.requires_grad
            )
            print(
                '{:}Starting training with a {:}simple Unet{:} '
                '({:}{:d}{:} parameters)'.format(
                    c['c'], c['g'], c['nc'], c['b'], n_param, c['nc']
                )
            )

        net.fit(
            train_loader, val_loader, test_loader, epochs=epochs,
            log_file=log_file
        )
        net.save_model(os.path.join(d_path, model_name))


def cross_val(
        d_path, n_folds=5, seed=42, loss='xent', verbose=0
):
    # Init
    c = color_codes()

    torch.manual_seed(seed)
    np.random.seed(seed)

    patient_dicts, n_images = get_images(d_path)

    n_patients = len(patient_dicts)

    for i in range(n_folds):
        print(
            '{:}Starting fold {:} {:}({:}) {:}'.format(
                c['c'], c['g'] + str(i) + c['nc'],
                c['y'], loss, c['nc'] + d_path
            )
        )
        # Training
        ini_test = n_patients * i // n_folds
        end_test = n_patients * (i + 1) // n_folds
        training = patient_dicts[end_test:] + patient_dicts[:ini_test]
        testing = patient_dicts[ini_test:end_test]

        net = SimpleUNet(n_images=n_images)

        model_name = 'unet-{:}.s{:d}.n{:d}.pt'.format(
            loss, seed, i
        )
        csv_name = 'unet-{:}.s{:d}.n{:d}.csv'.format(
            loss, seed, i
        )

        with open(os.path.join(d_path, csv_name), 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            train(
                d_path, net, model_name, training, testing,
                log_file=csvwriter, verbose=verbose
            )


"""
> Dummy main function
"""


def main(verbose=2):
    # Init
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    c = color_codes()
    options = parse_inputs()
    path_list = options['data_dir']

    print(
        '{:}[{:}] {:}<Segmentation pipeline>{:}'.format(
            c['c'], strftime("%H:%M:%S"), c['y'], c['nc']
        )
    )

    # Random seeds. These are hard coded to avoid issues if the script crashes.
    # A more elegant solution would be to create a "seed file", which is
    # instatiated on the first run and then it's checked if the script crashes.
    # Might do it "later".
    seeds = [42, 80702, 74794, 62021, 48497, 14813, 125, 44786, 21278, 26685]

    for d_path in path_list:
        for test_n, seed in enumerate(seeds):
            for loss in ['xent', 'gdsc', 'dsc', 'focal']:
                print(
                    '{:}[{:}] {:}Starting cross-validation {:d} - '
                    'seed {:d} {:}({:}){:}'.format(
                        c['c'], strftime("%H:%M:%S"), c['g'], test_n,
                        seed, c['y'], loss, c['nc']
                    )
                )
                cross_val(
                    d_path, seed=seed, loss=loss, verbose=verbose
                )


if __name__ == '__main__':
    main()
