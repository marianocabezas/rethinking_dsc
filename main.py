import csv
import time
import argparse
import os
import re
from time import strftime
import numpy as np
from nibabel import load as load_nii
import torch
from torch.utils.data import DataLoader
from utils import color_codes, get_dirs, get_int
from utils import get_mask, get_normalised_image
from utils import time_to_string, find_file
from models import SimpleUNet
from statistics import analyse_results
from datasets import LesionCroppingDataset, LesionDataset

"""
> Arguments
"""


def parse_inputs():
    # I decided to separate this function, for easier acces to the command line
    # parameters
    parser = argparse.ArgumentParser(
        description='Test different nets with 3D data.'
    )

    # Mode selector
    parser.add_argument(
        '-d', '--data-directory',
        dest='data_dir', default=[
            '/home/mariano/data/DiceProject/wmh',
            '/home/mariano/data/DiceProject/lit',
            '/home/mariano/data/DiceProject/msseg',
            '/home/mariano/data/DiceProject/longitudinal',
            '/home/mariano/data/DiceProject/cross-sectional',
            # '/home/mariano/data/DiceProject/enhancing',
        ],
        help='Option to define the folders for each ask with all the patients.'
    )
    parser.add_argument(
        '-e', '--epochs', dest='epochs',
        type=int, default=50,
        help='Number of epochs'
    )
    parser.add_argument(
        '-r', '--ratios', dest='ratios',
        type=int, nargs='+', default=[0, 1],
        help='Ratio of negative patches'
    )
    parser.add_argument(
        '-b', '--batch-size', dest='batch_size',
        type=int, default=8,
        help='Number of samples per batch'
    )
    parser.add_argument(
        '-p', '--patch-size', dest='patch_size',
        type=int, default=32,
        help='Patch size'
    )
    parser.add_argument(
        '-l', '--learning-rate',
        dest='lr',
        type=float, default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--sgd',
        dest='sgd', default=False, action='store_true',
        help='Option to use SGD instead of Adam'
    )
    parser.add_argument(
        '--batched',
        dest='batched', default=False, action='store_true',
        help='Option to run the main where every batch is used as validation'
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
        image_tags = ['t1', 'flair', 'pd', 't2']
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
        lesion = load_nii(os.path.join(p_path, 'lesion.nii.gz')).get_fdata()
        lesion = (lesion == 1).astype(np.uint8)
        images = [get_normalised_image(file, brain) for file in files]
        n_images = len(images)
        p_dict = {
            'name': p,
            'brain': brain,
            'lesion': lesion,
            'images': np.stack(images, axis=0),
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
        log_file=None, batch_file=None, verbose=0
):
    """

    :param d_path:
    :param net:
    :param model_name:
    :param train_dicts:
    :param test_dicts:
    :param negative_ratio:
    :param log_file:
    :param batch_file:
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

    if verbose > 1:
        print('Preparing the training datasets / dataloaders')

    # Here we'll do the training / validation split...
    d_train = [t['images'] for t in train_dicts]
    r_train = [t['brain'] for t in train_dicts]
    m_train = [t['lesion'] for t in train_dicts]
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
    val_dataset = LesionDataset(d_train, m_train, r_train)
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
        log_file=log_file, batch_file=batch_file
    )
    net.save_model(os.path.join(d_path, model_name))
    if batch_file is None:
        net.save_first(os.path.join(d_path, 'first_' + model_name))
        net.save_last(os.path.join(d_path, 'last_' + model_name))


"""
> Dummy main function
"""


def main(verbose=2):
    # Init
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    n_folds = 5
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
    # instantiated on the first run and then it's checked if the script end-s.
    # Might do it "later".
    seeds = [42, 80702, 74794, 62021, 48497]
    # losses = [
    #     'xent', 'xent_w', 'gdsc', 'gdsc_b', 'dsc', 'mixed', 'focal',
    #     'focal_w1', 'focal_w2', 'new'
    # ]
    losses = [
        'xent', 'xent_w', 'gdsc_b', 'dsc', 'mixed',
        'focal', 'focal_w1', 'focal_w2', 'new'
    ]
    optim = 'sgd' if options['sgd'] else 'adam'
    ratios = options['ratios']
    lr = options['lr']
    for d_path in path_list:
        for test_n, seed in enumerate(seeds):
            for nr in ratios:
                print(
                    '{:}[{:}] {:}Starting cross-validation {:d}{:}'.format(
                        c['c'], strftime("%H:%M:%S"), c['g'], test_n,
                        c['nc']
                    )
                )
                np.random.seed(seed)
                cross_seeds = np.random.randint(0, 10000, n_folds)

                patient_dicts, n_images = get_images(d_path)
                n_patients = len(patient_dicts)

                for i, seed_i in enumerate(cross_seeds):
                    for loss in losses:
                        np.random.seed(seed_i)
                        torch.manual_seed(seed_i)
                        print(
                            '{:}Starting fold {:} ({:}) {:} - {:d} '
                            '{:}[ratio {:d} - {:} lr {:.0e}]{:}'.format(
                                c['c'], c['g'] + str(i) + c['nc'] + c['y'],
                                loss, c['nc'] + d_path, seed, c['g'], nr,
                                optim, lr, c['nc']
                            )
                        )

                        model_name = 'unet-{:}.nr{:d}.s{:d}.n{:d}.' \
                                     '{:}-lr{:.0e}.pt'
                        model_name = model_name.format(
                            loss, nr, seed, i, optim, lr
                        )
                        net = SimpleUNet(
                            n_images=n_images, base_loss=loss,
                            lr=lr, optimiser=optim
                        )

                        try:
                            net.load_model(os.path.join(d_path, model_name))
                        except IOError:
                            # Training
                            # This weird sampling is mostly here for the
                            # MSSEG 2016 and WMH 2017 challenges, where we know
                            # how the different acquisitions are grouped. We
                            # want all the folds to have at least some images
                            # from each acquisition to make folds comparable.
                            # Otherwise the training and validation data, might
                            # not be the same and it might bias the curves.
                            test = list(range(i, n_patients, n_folds))
                            training = [
                                patient_dicts[t]
                                for t in range(len(patient_dicts))
                                if t not in test
                            ]
                            testing = [patient_dicts[t] for t in test]

                            csv_name = 'unet-{:}.nr{:d}.s{:d}.n{:d}' \
                                       '.{:}-lr{:.0e}.csv'

                            with open(
                                    os.path.join(d_path, csv_name.format(
                                        loss, nr, seed, i, optim, lr
                                    )), 'w'
                            ) as csvfile:
                                csvwriter = csv.writer(csvfile)
                                train(
                                    d_path, net, model_name, training, testing,
                                    negative_ratio=nr, log_file=csvwriter,
                                    verbose=verbose
                                )

        for nr in ratios:
            for loss in losses:
                analyse_results(
                    d_path, 'unet', loss, nr, list(range(n_folds)), lr
                )
                for i in range(n_folds):
                    analyse_results(d_path, 'unet', loss, nr, i, lr)


def batch_main(verbose=2):
    # Init
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    d_path = '/home/mariano/data/DiceProject/longitudinal'
    n_folds = 5
    c = color_codes()
    options = parse_inputs()

    print(
        '{:}[{:}] {:}<Segmentation pipeline>{:}'.format(
            c['c'], strftime("%H:%M:%S"), c['y'], c['nc']
        )
    )

    # Random seeds. These are hard coded to avoid issues if the script crashes.
    # A more elegant solution would be to create a "seed file", which is
    # instantiated on the first run and then it's checked if the script end-s.
    # Might do it "later".
    seeds = [42, 80702, 74794, 62021, 48497]
    # losses = [
    #     'xent', 'xent_w', 'gdsc', 'gdsc_b', 'dsc', 'mixed', 'focal',
    #     'focal_w1', 'focal_w2', 'new'
    # ]
    losses = [
        'xent', 'xent_w', 'gdsc_b', 'dsc', 'mixed', 'focal',
        'focal_w1', 'focal_w2', 'new'
    ]
    optim = 'sgd' if options['sgd'] else 'adam'
    ratios = options['ratios']
    lr = options['lr']
    for test_n, seed in enumerate(seeds):
        for nr in ratios:
            print(
                '{:}[{:}] {:}Starting batch cross-validation {:d}{:}'.format(
                    c['c'], strftime("%H:%M:%S"), c['g'], test_n,
                    c['nc']
                )
            )
            np.random.seed(seed)
            cross_seeds = np.random.randint(0, 10000, n_folds)

            patient_dicts, n_images = get_images(d_path)
            n_patients = len(patient_dicts)

            for i, seed_i in enumerate(cross_seeds):
                for loss in losses:
                    np.random.seed(seed_i)
                    torch.manual_seed(seed_i)
                    print(
                        '{:}Starting fold {:} ({:}) {:} - {:d} '
                        '{:}[ratio {:d} - {:} lr {:.0e}]{:}'.format(
                            c['c'], c['g'] + str(i) + c['nc'] + c['y'],
                            loss, c['nc'] + d_path, seed, c['g'], nr,
                            optim, lr, c['nc']
                        )
                    )

                    model_name = 'unet-batch-{:}.nr{:d}.s{:d}.n{:d}.' \
                                 '{:}-lr{:.0e}.pt'
                    model_name = model_name.format(
                        loss, nr, seed, i, optim, lr
                    )
                    net = SimpleUNet(
                        n_images=n_images, base_loss=loss,
                        lr=lr, optimiser=optim
                    )

                    try:
                        net.load_model(os.path.join(d_path, model_name))
                    except IOError:
                        # Training
                        test = list(range(i, n_patients, n_folds))
                        training = [
                            patient_dicts[t]
                            for t in range(len(patient_dicts))
                            if t not in test
                        ]
                        testing = [patient_dicts[t] for t in test]

                        csv_name = 'unet-batch-{:}.nr{:d}.s{:d}.n{:d}' \
                                   '.{:}-lr{:.0e}.csv'

                        with open(
                                os.path.join(d_path, csv_name.format(
                                    loss, nr, seed, i, optim, lr
                                )), 'w'
                        ) as csvfile:
                            csvwriter = csv.writer(csvfile)
                            train(
                                d_path, net, model_name, training, testing,
                                negative_ratio=nr, batch_file=csvwriter,
                                verbose=verbose
                            )

    for nr in ratios:
        for loss in losses:
            analyse_results(
                d_path, 'unet-batch', loss, nr, list(range(n_folds)), lr
            )
            for i in range(n_folds):
                analyse_results(d_path, 'unet-batch', loss, nr, i, lr)


if __name__ == '__main__':
    options = parse_inputs()
    if options['batched']:
        batch_main()
    else:
        main()
