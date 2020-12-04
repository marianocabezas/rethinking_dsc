import os
import time
import numpy as np
from nibabel import load as load_nii
from skimage.measure import label as bwlabeln
from utils import color_codes, get_dirs, get_int, time_to_string


def analyse_lesions(d_path, verbose=0):
    # Init
    c = color_codes()
    voxels = 0
    lesions = 0
    brain_voxels = 0
    patients = sorted(get_dirs(d_path))
    n_cases = len(patients)
    eval_start = time.time()

    if verbose > 0:
        print(''.join(['-'] * 58))
        print(
            '{:}{:^18}||{:^8}|{:^8}|{:^8}||{:^9}|'.format(
                c['clr'], 'Patient (t)', 'Vox', 'Lesions', '%', 'Brain'
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
        lesions_i = np.max(labels)
        lesions += lesions_i
        brain_voxels_i = np.sum(brain_mask)
        brain_voxels += brain_voxels_i

        if verbose > 1:
            print(
                '{:}{:<18}||{:8d}|{:8d}|{:7.4f}%||{:9d}|'.format(
                    c['clr'], patient,
                    voxels_i, lesions_i, 100 * voxels_i / brain_voxels_i,
                    brain_voxels_i
                )
            )
    if verbose > 0:
        print(
            '{:}{:<18}||{:8d}|{:8d}|{:7.4f}%||{:9d}|'.format(
                c['clr'], 'Mean',
                voxels, lesions, 100 * voxels / brain_voxels,
                brain_voxels
            )
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
