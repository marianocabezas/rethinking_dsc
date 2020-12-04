import os
import time
import numpy as np
from nibabel import load as load_nii
from skimage.measure import label as bwlabeln
from utils import color_codes, get_dirs, get_int, time_to_string


def analyse_lesions(d_path, verbose=0):
    # Init
    c = color_codes()
    patients = sorted(get_dirs(d_path))
    n_cases = len(patients)
    eval_start = time.time()

    if verbose > 0:
        print(''.join(['-'] * 89))
        print(
            '{:}{:^18}||{:^8}|{:^8}|{:^8}||{:^9}|'.format(
                c['clr'], 'Patient (t)', 'Vox', 'Lesions', '%', 'Brain'
            )
        )
    for i, (timepoint, patient) in enumerate(patients):
        image_path = os.path.join(d_path, patient, timepoint, 'preprocessed')
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
                '{:}Evaluating patient {:} [t = {:}] '
                '{:}[{:3d}/{:3d} - {:5.2f}%] {:} ETA: {:}'.format(
                    c['clr'], c['g'] + patient[:5] + c['nc'],
                    c['y'] + timepoint + c['nc'], c['c'],
                    i + 1, n_cases, 100 * (i + 1) / n_cases,
                    c['g'] + elapsed_s, eta_s + c['nc']
                ),
                end='\r'
            )

        cel_mask = load_nii(
            os.path.join(image_path, 'lesion.nii.gz')
        ).get_fdata().astype(np.bool)
        brain_mask = load_nii(
            os.path.join(image_path, 'brain.nii.gz')
        ).get_fdata().astype(np.bool)

        voxels_i = np.sum(cel_mask)
        labels = bwlabeln(cel_mask)
        lesions_i = np.max(labels)
        brain_voxels = np.sum(brain_mask)

        if verbose > 0:
            print(
                '{:}{:>7} ({:>8})||{:8d}|{:8d}|{:7.4f}%||{:9d}|'.format(
                    c['clr'], patient, timepoint,
                    voxels_i, lesions_i, 100 * voxels_i / brain_voxels,
                    brain_voxels
                )
            )

"""
> Dummy main function
"""


def main(verbose=2):
    path_list = [
        '/home/mariano/data/DiceProject/longitudinal',
        '/home/mariano/data/DiceProject/msseg',
        '/home/mariano/data/DiceProject/wmh',
        '/home/mariano/data/DiceProject/lit',
        '/home/mariano/data/DiceProject/cross-sectional',
    ]
    for d_path in path_list:
        analyse_lesions(d_path, verbose=verbose)


if __name__ == '__main__':
    main()
