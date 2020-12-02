import itertools
import numpy as np
from copy import deepcopy
from torch.utils.data.dataset import Dataset


''' Utility function for datasets '''


def get_bb(mask):
    """

    :param mask:
    :return:
    """
    idx = np.where(mask)
    bb = tuple(
        slice(min_i, max_i)
        for min_i, max_i in zip(
            np.min(idx, axis=-1), np.max(idx, axis=-1)
        )
    )
    return bb


def get_mask_voxels(mask):
    return map(tuple, np.stack(np.nonzero(mask), axis=1))


def centers_to_slice(voxels, patch_half):
    slices = [
        tuple(
            [
                slice(idx - p_len, idx + p_len) for idx, p_len in zip(
                    voxel, patch_half
                )
            ]
        ) for voxel in voxels
    ]
    return slices


def get_slices(masks, patch_size, overlap):
    """
    Function to get all the patches with a given patch size and overlap between
    consecutive patches from a given list of masks. We will only take patches
    inside the bounding box of the mask. We could probably just pass the shape
    because the masks should already be the bounding box.
    :param masks: List of masks.
    :param patch_size: Size of the patches.
    :param overlap: Overlap on each dimension between consecutive patches.

    """
    # Init
    # We will compute some intermediate stuff for later.
    patch_half = [p_length // 2 for p_length in patch_size]
    steps = [max(p_length - o, 1) for p_length, o in zip(patch_size, overlap)]

    # We will need to define the min and max pixel indices. We define the
    # centers for each patch, so the min and max should be defined by the
    # patch halves.
    min_bb = [patch_half] * len(masks)
    max_bb = [
        [
            max_i - p_len for max_i, p_len in zip(mask.shape, patch_half)
        ] for mask in masks
    ]

    # This is just a "pythonic" but complex way of defining all possible
    # indices given a min, max and step values for each dimension.
    dim_ranges = [
        map(
            lambda t: np.concatenate([np.arange(*t), [t[1]]]),
            zip(min_bb_i, max_bb_i, steps)
        ) for min_bb_i, max_bb_i in zip(min_bb, max_bb)
    ]

    # And this is another "pythonic" but not so intuitive way of computing
    # all possible triplets of center voxel indices given the previous
    # indices. I also added the slice computation (which makes the last step
    # of defining the patches).
    patch_slices = [
        centers_to_slice(
            itertools.product(*dim_range), patch_half
        ) for dim_range in dim_ranges
    ]

    return patch_slices


''' Datasets '''


class LesionCroppingDataset(Dataset):
    """
    This is a training dataset and we only want patches that
    actually have lesions since there are lots of non-lesion voxels
    anyways.
    """
    def __init__(
            self,
            cases, labels, masks, patch_size=32, overlap=0, balanced=True,
            negative_ratio=1
    ):
        # Init
        data_shape = masks[0].shape
        if type(patch_size) is not tuple:
            self.patch_size = (patch_size,) * len(data_shape)
        else:
            self.patch_size = patch_size
        if type(overlap) is not tuple:
            self.overlap = (overlap,) * len(data_shape)
        else:
            self.overlap = overlap
        self.balanced = balanced

        self.masks = masks
        self.labels = labels
        self.cases = cases

        self.ratio = negative_ratio

        # We get the preliminary patch slices (inside the bounding box)...
        slices = get_slices(self.masks, self.patch_size, self.overlap)

        # ... however, being inside the bounding box doesn't guarantee that the
        # patch itself will contain any lesion voxels. Since, the lesion class
        # is extremely underrepresented, we will filter this preliminary slices
        # to guarantee that we only keep the ones that contain at least one
        # lesion voxel.
        if self.balanced:
            self.positive_slices = [
                (s, i)
                for i, (label, s_i) in enumerate(zip(self.labels, slices))
                for s in s_i if np.sum(label[s]) > 0
            ]
            self.negative_slices = [
                (s, i)
                for i, (label, mask, s_i) in enumerate(
                    zip(self.labels, self.masks, slices)
                )
                for s in s_i if (np.sum(label[s]) == 0) & (np.sum(mask[s]) > 0)
            ]
            self.current_negative = deepcopy(self.negative_slices)
        else:
            self.patch_slices = [
                (s, i) for i, s_i in enumerate(slices) for s in s_i
            ]

    def __getitem__(self, index):
        flip = False
        if self.balanced:
            if index < len(self.positive_slices) * 2:
                if index < len(self.positive_slices):
                    slice_i, case_idx = self.positive_slices[index]
                else:
                    flip = True
                    index -= len(self.positive_slices)
                    slice_i, case_idx = self.positive_slices[index]
            else:
                index = np.random.randint(len(self.current_negative))
                slice_i, case_idx = self.current_negative.pop(index)
                if len(self.current_negative) == 0:
                    self.current_negative = deepcopy(self.negative_slices)
        else:
            slice_i, case_idx = self.patch_slices[index]

        case = self.cases[case_idx]
        none_slice = (slice(None, None),)
        # Patch "extraction".
        print(case.shape)
        data = case[none_slice + slice_i].astype(np.float32)
        labels = self.labels[case_idx].astype(np.uint8)

        # We expand the labels to have 1 "channel". This is tricky depending
        # on the loss function (some require channels, some don't).
        target = np.expand_dims(labels[slice_i], 0)

        if flip:
            data = np.fliplr(data).copy()
            target = np.fliplr(target).copy()

        return data, target

    def __len__(self):
        if self.balanced:
            positive_samples = len(self.positive_slices)
            return positive_samples * 2 + positive_samples * 2 * self.ratio
        else:
            return len(self.patch_slices)


class LesionDataset(Dataset):
    """
    This is a training dataset and we only want patches that
    actually have lesions since there are lots of non-lesion voxels
    anyways.
    """
    def __init__(
            self,
            cases, labels, masks
    ):
        # Init
        self.masks = masks
        self.labels = labels
        self.cases = cases

    def __getitem__(self, index):
        bb = get_bb(self.masks[index])
        data = self.cases[index][(slice(None),) + bb].astype(np.float32)
        labels = self.labels[index][bb].astype(np.uint8)

        # We expand the labels to have 1 "channel". This is tricky depending
        # on the loss function (some require channels, some don't).
        target = np.expand_dims(labels, 0)

        return data, target

    def __len__(self):
        return len(self.cases)
