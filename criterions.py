import torch
from torch.nn import functional as F


"""
Probabilistic losses (training)
"""


def focal_loss(pred, target, alpha=0.2, gamma=2.0):
    """
    Function to compute the focal loss based on:
    Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár. "Focal
    Loss for Dense Object Detection".
    https://arxiv.org/abs/1708.02002
    https://ieeexplore.ieee.org/document/8237586
    :param pred: Predicted values. The shape of the tensor should be:
     [n_batches, data_shape]
    :param target: Ground truth values. The shape of the tensor should be:
     [n_batches, data_shape]
    :param alpha: Weighting parameter to avoid class imbalance (default 0.2).
    :param gamma: Focusing parameter (default 2.0).
    :return: Focal loss value.
    """

    m_bg = target == 0
    m_fg = target > 0

    alpha_fg = alpha
    alpha_bg = 1 - alpha
    pt_fg = pred[m_fg]
    pt_bg = (1 - pred[m_fg])

    bce = F.binary_cross_entropy(pred, target, reduction='none')
    bce_fg = bce[m_fg]
    bce_bg = bce[m_bg]

    focal_fg = alpha_fg * (1 - pt_fg).pow(gamma) * bce_fg
    focal_bg = alpha_bg * (1 - pt_bg).pow(gamma) * bce_bg

    focal = torch.cat([focal_fg, focal_bg])

    # pt = target.type_as(pred) * pred + (1 - target).type_as(pred) * (1 - pred)
    # bce = F.binary_cross_entropy(pred, target, reduction='none')
    # focal = alpha * (1 - pt).pow(gamma) * bce
    return focal.mean()


def gendsc_loss(pred, target, batch=True, w_bg=None, w_fg=None):
    """
    Function to compute the generalised Dice loss based on:
    Carole H. Sudre, Wenqi Li, Tom Vercauteren, Sebastien Ourselin, M. Jorge
    Cardoso, "Generalised Dice Overlap as a Deep Learning Loss Function for
    Highly Unbalanced Segmentations".
    https://arxiv.org/abs/1707.03237
    https://link.springer.com/chapter/10.1007/978-3-319-67558-9_28
    :param pred: Predicted values. This tensor should have the shape:
     [batch_size, data_shape]
    :param target: Ground truth values. This tensor should have the shape:
     [batch_size, data_shape]
    :param w_bg: Weight given to background voxels.
    :param w_fg: Weight given to foreground voxels.
    :return: The DSC loss for the batch
    """
    # Init
    # Dimension checks. We want everything to be the same. This a class vs
    # class comparison.
    assert target.shape == pred.shape,\
        'Sizes between predicted and target do not match'
    target = target.type_as(pred)

    # Mj = Labels of patch j / mij = Label of voxel i from patch j
    # Aj = prediction of patch j / aij = Prediction of voxel i from patch j
    # M0 = {i: mij = 0}
    # M1 = {i: mij = 1}
    # Ldsc = Ldsc(A, M)) =
    # = sum(||M1|| + epsilon + sum^M0(ak0) - sum^M1(ak1)) /
    # ||M1|| + epsilon + sum(ai)

    if batch:
        loss = []
        for y, y_hat in zip(target, pred):
            m_bg = y == 0
            m_fg = y > 0
            n_bg = torch.sum(m_bg)
            n_fg = torch.sum(m_fg)
            n = torch.numel(y)

            if w_bg is None:
                if n_bg > 0:
                    w_bg = torch.sqrt(n_bg.type(torch.float32)) ** -2
                else:
                    w_bg = 0
            if w_fg is None:
                if n_fg > 0:
                    w_fg = torch.sqrt(n_fg.type(torch.float32)) ** -2
                else:
                    w_fg = 0

            sum_pred_fg = torch.sum(y_hat[m_fg])
            sum_pred = torch.sum(y_hat)

            tp_term = (w_fg + w_bg) * sum_pred_fg
            tn_term = w_bg * (n_bg - sum_pred)
            den = (w_fg - w_bg) * (n_fg + sum_pred) + 2 * n * w_bg

            loss.append(1 - 2 * (tp_term + tn_term) / den)
        loss = torch.mean(torch.tensor(loss))
    else:
        m_bg = target == 0
        m_fg = target > 0
        n_bg = torch.sum(m_bg)
        n_fg = torch.sum(m_fg)
        n = torch.numel(target)

        if w_bg is None:
            if n_bg > 0:
                w_bg = torch.sqrt(n_bg) ** -2
            else:
                w_bg = 0
        if w_fg is None:
            if n_fg > 0:
                w_fg = torch.sqrt(n_bg) ** -2
            else:
                w_fg = 0

        sum_pred_fg = torch.sum(pred[m_fg])
        sum_pred = torch.sum(pred)

        tp_term = (w_fg + w_bg) * sum_pred_fg
        tn_term = w_bg * (n_bg - sum_pred)
        den = (w_fg - w_bg) * (n_fg + sum_pred) + 2 * n * w_bg

        loss = 1 - 2 * (tp_term + tn_term) / den

    return loss


"""
Binary losses (validation)
"""


def dsc_binary_loss(pred, target):
    pred = (pred > 0.5).to(pred.device)
    target = target.type_as(pred).to(pred.device)

    dims = pred.shape
    reduce_dims = tuple(range(1, len(dims)))
    intersection = (
            2 * torch.sum(pred & target, dim=reduce_dims)
    ).type(torch.float32).to(pred.device)
    sum_pred = torch.sum(
        pred, dim=reduce_dims
    ).type(torch.float32).to(pred.device)
    sum_target = torch.sum(
        target, dim=reduce_dims
    ).type(torch.float32).to(pred.device)

    dsc_k = intersection / (sum_pred + sum_target)
    dsc_k[torch.isnan(dsc_k)] = 0
    dsc = 1 - torch.mean(dsc_k)

    return torch.clamp(dsc, 0., 1.)


def tp_binary_loss(pred, target):
    pred = (pred >= 0.5).to(pred.device)
    target = target.type_as(pred).to(pred.device)

    dims = pred.shape
    reduce_dims = tuple(range(1, len(dims)))
    intersection = (
            torch.sum(pred & target, dim=reduce_dims)
    ).type(torch.float32).to(pred.device)
    sum_target = torch.sum(
        target, dim=reduce_dims
    ).type(torch.float32).to(pred.device)

    tp_k = intersection / sum_target
    tp_k[torch.isnan(tp_k)] = 0
    tp = 1 - torch.mean(tp_k)

    return torch.clamp(tp, 0., 1.)


def tn_binary_loss(pred, target):
    pred = (pred < 0.5).to(pred.device)
    target = torch.logical_not(target.type_as(pred)).to(pred.device)

    dims = pred.shape
    reduce_dims = tuple(range(1, len(dims)))
    intersection = (
            torch.sum(pred & target, dim=reduce_dims)
    ).type(torch.float32).to(pred.device)
    sum_target = torch.sum(
        target, dim=reduce_dims
    ).type(torch.float32).to(pred.device)

    tn_k = intersection / sum_target
    tn_k[torch.isnan(tn_k)] = 0
    tn = 1 - torch.mean(tn_k)

    return torch.clamp(tn, 0., 1.)
