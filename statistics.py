import csv
import os
import time
from operator import and_
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
from nibabel import load as load_nii
from skimage.measure import label as bwlabeln
import torch
from torch.utils.data import DataLoader
from datasets import LesionCroppingDataset
from main import get_images, parse_inputs
from utils import color_codes, get_dirs, get_int, time_to_string


def analyse_lesions(d_path, verbose=0):
    # Init
    c = color_codes()
    batch_percent = []
    voxel_percent = []
    voxels = 0
    batch_voxels = 0
    lesions = 0
    brain_voxels = 0
    batch_brain_voxels = 0
    patients = sorted(get_dirs(d_path))
    n_cases = len(patients)
    eval_start = time.time()

    if verbose > 1:
        print(''.join(['-'] * 58))
        print(
            '{:}{:^18}||{:^8}|{:^8}|{:^8}||{:^9}|'.format(
                c['clr'], 'Patient', 'Vox', 'Lesions', '%', 'Brain'
            )
        )
    for i, patient in enumerate(patients):
        image_path = os.path.join(d_path, patient)
        if verbose > 0:
            t_elapsed = time.time() - eval_start
            elapsed_s = time_to_string(t_elapsed)
            if i > 0:
                t_step = t_elapsed / i
            else:
                t_step = 0
            steps_left = n_cases - i
            eta_s = time_to_string(steps_left * t_step)
            print(
                '{:}Evaluating patient {:}[{:3d}/{:3d} - {:5.2f}%]'
                ' {:} ETA: {:}'.format(
                    c['clr'], c['g'] + patient[:5] + c['nc'] + c['c'],
                    i + 1, n_cases, 100 * (i + 1) / n_cases,
                    c['g'] + elapsed_s, eta_s + c['nc']
                ),
                end='\r'
            )

        lesion_mask = load_nii(
            os.path.join(image_path, 'lesion.nii.gz')
        ).get_fdata().astype(np.bool)
        brain_mask = load_nii(
            os.path.join(image_path, 'brain.nii.gz')
        ).get_fdata().astype(np.bool)

        voxels_i = np.sum(lesion_mask)
        voxels += voxels_i
        labels = bwlabeln(lesion_mask)
        lesions_i = len(np.unique(labels)) - 1
        lesions += lesions_i
        brain_voxels_i = np.sum(brain_mask)
        brain_voxels += brain_voxels_i
        voxel_percent.append(100 * voxels_i / brain_voxels_i)

        if verbose > 1:
            print(
                '{:}{:<18}||{:8d}|{:8d}|{:7.4f}%||{:9d}|'.format(
                    c['clr'], patient,
                    voxels_i, lesions_i, 100 * voxels_i / brain_voxels_i,
                    brain_voxels_i
                )
            )

    options = parse_inputs()
    batch_size = options['batch_size']
    patch_size = options['patch_size']
    overlap = patch_size // 2

    train_dicts, _ = get_images(d_path)
    # Here we'll do the training / validation split...
    d_train = [t['images'] for t in train_dicts]
    r_train = [t['brain'] for t in train_dicts]
    m_train = [t['lesion'] for t in train_dicts]

    # Training
    if verbose > 1:
        print('< Training dataset >')
    train_dataset = LesionCroppingDataset(
        d_train, m_train, r_train, patch_size=patch_size,
        overlap=overlap, negative_ratio=0
    )
    train_loader = DataLoader(train_dataset, batch_size, True)

    for brain, y in train_loader:
        y_flat = y.flatten(1)
        b_flat = brain.flatten(1)
        batch_voxels += torch.sum(y)
        batch_brain_voxels += torch.sum(brain)
        percent = 100 * torch.sum(y_flat, dim=1) / torch.sum(b_flat, dim=1)
        batch_percent + percent.numpy().tolst()

    if verbose > 0:
        print(''.join(['-'] * 86))
        print(
            '{:}{:^18}||{:^8}|{:^8}||{:^27}|{:^27}||{:^9}||{:^9}'.format(
                c['clr'], 'Patient', 'Vox', 'Lesions', '% image',
                '% batch', 'Brain', 'Size'
            )
        )
        print(
            '{:}{:<18}||{:8d}|{:8d}|'
            '|{:7.4f}±{:7.4f}% [{:8.4f}%]|{:7.4f}±{:7.4f}% [{:8.4f}%]|'
            '|{:9d}||{:8.4f}'.format(
                c['clr'], 'Mean',
                voxels, lesions, np.mean(voxel_percent), np.std(voxel_percent),
                100 * voxels / brain_voxels,
                np.mean(batch_percent), np.std(batch_percent),
                100 * batch_voxels / batch_brain_voxels,
                brain_voxels, voxels / lesions
            )
        )
        print(''.join(['-'] * 86))


def check_tags(filename, tags):
    check = [tag in filename for tag in tags]
    return reduce(and_, check)


def save_bands(
        x, y, yinf, ysup, suffix, path,
        xmin=None, xmax=None, ymin=0, ymax=1,
        xlabel='Epoch', ylabel='Metric', legend=None
):
    # Init
    if xmin is None:
        xmin = np.min(x)
    if xmax is None:
        xmax = np.max(x)
    if ymin is None:
        ymin = np.min(yinf)
    if ymax is None:
        ymax = np.max(ysup)

    fig = plt.figure()
    ax = fig.add_subplot()
    plt.title('Band plot (train vs validation vs test)'.format(xlabel, ylabel))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    colomap = ['b', 'g', 'c', 'r', 'm', 'y']

    for yi, yinfi, ysupi, ci in zip(y, yinf, ysup, colomap):
        ax.plot(x, yi, '-', color=ci)
        ax.fill_between(x, yinfi, ysupi, alpha=0.2, color=ci)

    ax.set_xlim(xmin=xmin, xmax=xmax)
    ax.set_ylim(ymin=ymin, ymax=ymax)

    if legend is not None:
        plt.legend(legend)

    plt.savefig(os.path.join(
        path, 'bands_{:}.png'.format(suffix)
    ))
    plt.close()


def analyse_results(path, net, loss, ratio, fold, lr):
    if isinstance(fold, list):
        tags = [
            '{:}-{:}.'.format(net, loss), 'nr{:d}'.format(ratio),
            'lr{:.0e}'.format(lr)
        ]
        suffix = '{:}.nr{:d}.lr{:.0e}'.format(loss, ratio, lr)
    else:
        tags = [
            'unet-{:}.'.format(loss), 'nr{:d}'.format(ratio),
            'n{:d}'.format(fold), 'lr{:.0e}'.format(lr)
        ]
        suffix = '{:}.n{:d}.nr{:d}.lr{:.0e}'.format(loss, fold, ratio, lr)
    csv_files = [
        file for file in os.listdir(path)
        if file.endswith('.csv') and check_tags(file, tags)
    ]
    dicts = []
    keys = []
    for file in csv_files:
        csv_name = os.path.join(path, file)
        with open(csv_name) as csv_file:
            reader = csv.reader(csv_file)
            keys = next(reader)
            results_dict = {}
            for key in keys[1:-1]:
                results_dict[key] = []
            for row in reader:
                for key, data in zip(keys[1:-1], row[1:-1]):
                    results_dict[key].append(float(data))
        dicts.append(results_dict)

    final_dict = {}
    for key in keys[1:-1]:
        metrics = np.stack([metrics[key] for metrics in dicts], axis=0)
        final_dict['min_' + key] = np.min(metrics, axis=0)
        final_dict['mean_' + key] = np.mean(metrics, axis=0)
        final_dict['max_' + key] = np.max(metrics, axis=0)

    x = range(len(final_dict['min_train_dsc']))
    y_dsc = [
        final_dict['mean_train_dsc'], final_dict['mean_val_dsc'],
        final_dict['mean_test_dsc']
    ]
    y_fn = [
        final_dict['mean_train_fn'], final_dict['mean_val_fn'],
        final_dict['mean_test_fn'],
    ]
    yinf_dsc = [
        final_dict['min_train_dsc'], final_dict['min_val_dsc'],
        final_dict['min_test_dsc']
    ]
    yinf_fn = [
        final_dict['min_train_fn'], final_dict['min_val_fn'],
        final_dict['min_test_fn']
    ]
    ysup_dsc = [
        final_dict['max_train_dsc'], final_dict['max_val_dsc'],
        final_dict['max_test_dsc']
    ]
    ysup_fn = [
        final_dict['max_train_fn'], final_dict['max_val_fn'],
        final_dict['max_test_fn']
    ]
    save_bands(
        x, y_dsc, yinf_dsc, ysup_dsc, '{:}-dsc'.format(suffix),
        path, ymin=0, ymax=1, ylabel='DSC metric', legend=[
            'Patch training DSC', 'Image training DSC', 'Image testing DSC'
        ]
    )
    save_bands(
        x, y_fn, yinf_fn, ysup_fn, '{:}-fn'.format(suffix),
        path, ymin=0, ymax=1, ylabel='FN metric', legend=[
            'Patch training FN', 'Image training FN', 'Image testing FN'
        ]
    )
    save_bands(
        x, y_dsc + y_fn, yinf_dsc + yinf_fn, ysup_dsc + ysup_fn,
        suffix, path, ymin=0, ymax=1,
        legend=[
            'Patch training DSC', 'Image training DSC', 'Image testing DSC',
            'Patch training FN', 'Image training FN', 'Image testing FN'
        ]
    )


"""
> Dummy main function
"""


def main(verbose=1):
    # Init
    c = color_codes()
    path_list = [
        '/home/mariano/data/DiceProject/longitudinal',
        '/home/mariano/data/DiceProject/msseg',
        '/home/mariano/data/DiceProject/wmh',
        '/home/mariano/data/DiceProject/lit',
        '/home/mariano/data/DiceProject/cross-sectional',
    ]
    for d_path in path_list:
        print(''.join(['-'] * 58))
        print('{:}{:^57}{:}|'.format(c['clr'] + c['g'], d_path, c['nc']))
        analyse_lesions(d_path, verbose=verbose)
    print(''.join(['-'] * 58))


if __name__ == '__main__':
    main()
