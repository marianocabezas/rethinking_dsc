import time
import itertools
from functools import partial
from copy import deepcopy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import time_to_string, to_torch_var
from criterions import dsc_binary_loss, tp_binary_loss, tn_binary_loss
from criterions import focal_loss, gendsc_loss, new_loss


def norm_f(n_f):
    return nn.GroupNorm(n_f // 8, n_f)


class BaseModel(nn.Module):
    """"
    This is the baseline model to be used for any of my networks. The idea
    of this model is to create a basic framework that works similarly to
    keras, but flexible enough.
    For that reason, I have "embedded" the typical pytorch main loop into a
    fit function and I have defined some intermediate functions and callbacks
    to alter the main loop. By itself, this model can train any "normal"
    network with different losses and scores for training and validation.
    It can be easily extended to create adversarial networks (which I have done
    in other repositories) and probably to other more complex problems.
    The network also includes some print functions to check the current status.
    """
    def __init__(self):
        """
        Main init. By default some parameters are set, but they should be
        redefined on networks inheriting that model.
        """
        super().__init__()
        # Init values
        self.init = True
        self.optimizer_alg = None
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.dropout = 0
        self.final_dropout = 0
        self.ann_rate = 0
        self.best_loss_tr = np.inf
        self.best_loss_val = np.inf
        self.best_state = None
        self.best_opt = None
        self.train_functions = [
            {'name': 'train', 'weight': 1, 'f': None},
        ]
        self.val_functions = [
            {'name': 'val', 'weight': 1, 'f': None},
        ]
        self.acc_functions = {}
        self.acc = None
        self.first_state = None
        self.last_state = None

    def forward(self, *inputs):
        """

        :param inputs: Inputs to the forward function. We are passing the
         contents by reference, so if there are more than one input, they
         will be separated.
        :return: Nothing. This has to be reimplemented for any class.
        """
        return None

    def mini_batch_loop(
            self, data, train=True
    ):
        """
        This is the main loop. It's "generic" enough to account for multiple
        types of data (target and input) and it differentiates between
        training and testing. While inherently all networks have a training
        state to check, here the difference is applied to the kind of data
        being used (is it the validation data or the training data?). Why am
        I doing this? Because there might be different metrics for each type
        of data. There is also the fact that for training, I really don't care
        about the values of the losses, since I only want to see how the global
        value updates, while I want both (the losses and the global one) for
        validation.
        :param data: Dataloader for the network.
        :param train: Whether to use the training dataloader or the validation
         one.
        :return:
        """
        losses = list()
        mid_losses = list()
        n_batches = len(data)
        for batch_i, (x, y) in enumerate(data):
            # In case we are training the the gradient to zero.
            if self.training:
                self.optimizer_alg.zero_grad()

            # First, we do a forward pass through the network.
            if isinstance(x, list) or isinstance(x, tuple):
                x_cuda = tuple(x_i.to(self.device) for x_i in x)
                pred_labels = self(*x_cuda)
            else:
                pred_labels = self(x.to(self.device))
            if isinstance(y, list) or isinstance(y, tuple):
                y_cuda = tuple(y_i.to(self.device) for y_i in y)
            else:
                y_cuda = y.to(self.device)

            # After that, we can compute the relevant losses.
            if train:
                # Training losses (applied to the training data)
                batch_losses = [
                    l_f['weight'] * l_f['f'](pred_labels, y_cuda)
                    for l_f in self.train_functions
                ]
                batch_loss = sum(batch_losses)
                if self.training:
                    batch_loss.backward()
                    self.optimizer_alg.step()
                    self.batch_update(len(data))

            else:
                # Validation losses (applied to the validation data)
                batch_losses = [
                    l_f['f'](pred_labels, y_cuda)
                    for l_f in self.val_functions
                ]
                batch_loss = sum([
                    l_f['weight'] * l
                    for l_f, l in zip(self.val_functions, batch_losses)
                ])
                mid_losses.append([loss.tolist() for loss in batch_losses])

            # It's important to compute the global loss in both cases.
            loss_value = batch_loss.tolist()
            losses.append(loss_value)

            # Curriculum dropout / Adaptive dropout
            # Here we could modify dropout to be updated for each batch.
            # (1 - rho) * exp(- gamma * t) + rho, gamma > 0

            self.print_progress(
                batch_i, n_batches, loss_value, np.mean(losses)
            )

        # Mean loss of the global loss (we don't need the loss for each batch).
        mean_loss = np.mean(losses)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        if train:
            return mean_loss
        else:
            # If using the validation data, we actually need to compute the
            # mean of each different loss.
            mean_losses = np.mean(list(zip(*mid_losses)), axis=1)
            return mean_loss, mean_losses

    def fit(
            self,
            train_loader,
            val_loader,
            test_loader,
            epochs=100,
            log_file=None,
            verbose=True
    ):
        # Init
        self.first_state = deepcopy(self.state_dict())
        best_e = 0
        l_names = ['train', ' val '] + [
            '{:^6s}'.format('vl_' + l_f['name'][:3])
            for l_f in self.val_functions
        ] + [
            '{:^6s}'.format('ts_' + l_f['name'][:3])
            for l_f in self.val_functions
        ]
        l_bars = '--|--'.join(
            ['-' * 5] * 2 +
            ['-' * 6] * len(l_names[2:])
        )
        l_hdr = '  |  '.join(l_names)
        # Since we haven't trained the network yet, we'll assume that the
        # initial values are the best ones.
        self.best_state = deepcopy(self.state_dict())
        self.best_opt = deepcopy(self.optimizer_alg.state_dict())
        t_start = time.time()

        # We'll just take the maximum losses and accuracies (inf, -inf)
        # and print the headers.
        print('\033[K', end='')
        print('Epoch num |  {:}  |'.format(l_hdr))
        print('----------|--{:}--|'.format(l_bars))
        best_loss_tr = [np.inf] * len(self.val_functions)
        best_loss_val = [np.inf] * len(self.val_functions)
        best_loss_tst = [np.inf] * len(self.val_functions)

        if log_file is not None:
            log_file.writerow(
                ['Epoch', 'train', 'val'] + [
                    'train_' + l_f['name']
                    for l_f in self.val_functions
                ] + [
                    'val_' + l_f['name']
                    for l_f in self.val_functions
                ] + [
                    'test_' + l_f['name']
                    for l_f in self.val_functions
                ] + ['time']
            )

        for self.epoch in range(epochs):
            # Main epoch loop
            self.t_train = time.time()
            self.train()
            # First we train and check if there has been an improvement.
            loss_tr = self.mini_batch_loop(train_loader)
            improvement_tr = self.best_loss_tr > loss_tr
            if improvement_tr:
                self.best_loss_tr = loss_tr
                tr_loss_s = '\033[32m{:7.4f}\033[0m'.format(loss_tr)
            else:
                tr_loss_s = '{:7.4f}'.format(loss_tr)

            # Then we validate and check all the losses
            _, best_loss_tr, _, mid_tr = self.validate(
                train_loader, best_loss_tr
            )

            loss_val, best_loss_val, losses_val_s, mid_val = self.validate(
                val_loader, best_loss_val
            )

            _, best_loss_tst, losses_tst_s, mid_tst = self.validate(
                test_loader, best_loss_tst
            )

            # Patience check
            # We check the patience to stop early if the network is not
            # improving. Otherwise we are wasting resources and time.
            improvement_val = self.best_loss_val > loss_val
            loss_s = '{:7.4f}'.format(loss_val)
            if improvement_val:
                self.best_loss_val = loss_val
                epoch_s = '\033[32mEpoch {:03d}\033[0m'.format(self.epoch)
                loss_s = '\033[32m{:}\033[0m'.format(loss_s)
                best_e = self.epoch
                self.best_state = deepcopy(self.state_dict())
                self.best_opt = deepcopy(self.optimizer_alg.state_dict())
            else:
                epoch_s = 'Epoch {:03d}'.format(self.epoch)

            t_out = time.time() - self.t_train
            t_s = time_to_string(t_out)

            if verbose:
                print('\033[K', end='')
                final_s = ' | '.join(
                    [epoch_s, tr_loss_s, loss_s] +
                    losses_val_s + losses_tst_s +
                    [t_s]
                )
                print(final_s)
            if log_file is not None:
                log_file.writerow(
                    [
                        'Epoch {:03d}'.format(self.epoch),
                        '{:7.4f}'.format(loss_tr),
                        '{:7.4f}'.format(loss_val)
                    ] + mid_tr.tolist() + mid_val.tolist() + mid_tst.tolist() +
                    [t_s]
                )

            self.epoch_update(epochs)

        self.last_state = deepcopy(self.state_dict())
        self.epoch = best_e
        self.load_state_dict(self.best_state)
        t_end = time.time() - t_start
        t_end_s = time_to_string(t_end)
        if verbose:
            print(
                    'Training finished in {:} epochs ({:}) '
                    'with minimum validation loss = {:f} '
                    '(epoch {:03d})'.format(
                        self.epoch + 1, t_end_s, self.best_loss_val, best_e
                    )
            )

    def validate(self, data, best_loss):
        with torch.no_grad():
            self.t_val = time.time()
            self.eval()
            loss, mid_losses = self.mini_batch_loop(
                data, False
            )
        # Mid losses check
        losses_s = [
            '\033[36m{:8.5f}\033[0m'.format(l) if bl > l
            else '{:8.5f}'.format(l) for bl, l in zip(
                best_loss, mid_losses
            )
        ]
        best_loss = [
            l if bl > l else bl for bl, l in zip(
                best_loss, mid_losses
            )
        ]

        return loss, best_loss, losses_s, mid_losses

    def epoch_update(self, epochs):
        """
        Callback function to update something on the model after the epoch
        is finished. To be reimplemented if necessary.
        :param epochs: Maximum number of epochs
        :return: Nothing.
        """
        return None

    def batch_update(self, batches):
        """
        Callback function to update something on the model after the batch
        is finished. To be reimplemented if necessary.
        :param batches: Maximum number of epochs
        :return: Nothing.
        """
        return None

    def print_progress(self, batch_i, n_batches, b_loss, mean_loss):
        """
        Function to print the progress of a batch. It takes into account
        whether we are training or validating and uses different colors to
        show that. It's based on Keras arrow progress bar, but it only shows
        the current (and current mean) training loss, elapsed time and ETA.
        :param batch_i: Current batch number.
        :param n_batches: Total number of batches.
        :param b_loss: Current loss.
        :param mean_loss: Current mean loss.
        :return: None.
        """
        init_c = '\033[0m' if self.training else '\033[38;5;238m'
        percent = 20 * (batch_i + 1) // n_batches
        progress_s = ''.join(['-'] * percent)
        remainder_s = ''.join([' '] * (20 - percent))
        loss_name = 'train_loss' if self.training else 'val_loss'

        if self.training:
            t_out = time.time() - self.t_train
        else:
            t_out = time.time() - self.t_val
        time_s = time_to_string(t_out)

        t_eta = (t_out / (batch_i + 1)) * (n_batches - (batch_i + 1))
        eta_s = time_to_string(t_eta)
        epoch_hdr = '{:}Epoch {:03} ({:03d}/{:03d}) [{:}>{:}] '
        loss_s = '{:} {:f} ({:f}) {:} / ETA {:}'
        batch_s = (epoch_hdr + loss_s).format(
            init_c, self.epoch, batch_i + 1, n_batches,
            progress_s, remainder_s,
            loss_name, b_loss, mean_loss, time_s, eta_s + '\033[0m'
        )
        print('\033[K', end='', flush=True)
        print(batch_s, end='\r', flush=True)

    def freeze(self):
        """
        Method to freeze all the network parameters.
        :return: None
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Method to unfreeze all the network parameters.
        :return: None
        """
        for param in self.parameters():
            param.requires_grad = True

    def save_model(self, net_name):
        torch.save(self.state_dict(), net_name)

    def save_last(self, net_name):
        if self.last_state is not None:
            torch.save(self.last_state, net_name)

    def save_first(self, net_name):
        if self.first_state is not None:
            torch.save(self.first_state, net_name)

    def load_model(self, net_name):
        self.load_state_dict(torch.load(net_name))


class Autoencoder(BaseModel):
    """
    Main autoencoder class. This class can actually be parameterised on init
    to have different "main blocks", normalisation layers and activation
    functions.
    """
    def __init__(
            self,
            conv_filters,
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
            n_inputs=1,
            kernel=3,
            pooling=False,
            norm=None,
            activation=None,
            block=None,
            dropout=0,
    ):
        """
        Constructor of the class. It's heavily parameterisable to allow for
        different autoencoder setups (residual blocks, double convolutions,
        different normalisation and activations).
        :param conv_filters: Filters for both the encoder and decoder. The
         decoder mirrors the filters of the encoder.
        :param device: Device where the model is stored (default is the first
         cuda device).
        :param n_inputs: Number of input channels.
        :param kernel: Kernel width for the main block.
        :param pooling: Whether to use pooling or not.
        :param norm: Normalisation block (it has to be a pointer to a valid
         normalisation Module).
        :param activation: Activation block (it has to be a pointer to a valid
         activation Module).
        :param block: Main block. It has to be a pointer to a valid block from
         this python file (otherwise it will fail when trying to create a
         partial of it).
        :param dropout: Dropout value.
        """
        super().__init__()
        # Init
        if norm is None:
            norm = partial(lambda ch_in: nn.Sequential())
        if block is None:
            block = Conv3dBlock
        block_partial = partial(
            block, kernel=kernel, norm=norm, activation=activation
        )
        self.n_inputs = n_inputs
        self.pooling = pooling
        self.device = device
        self.dropout = dropout
        self.filters = conv_filters

        conv_in, conv_out, deconv_in, deconv_out = self.compute_filters()

        # Down path
        # We'll use the partial and fill it with the channels for input and
        # output for each level.
        self.down = nn.ModuleList([
            block_partial(f_in, f_out) for f_in, f_out in zip(
                conv_in, conv_out
            )
        ])

        # Bottleneck
        self.u = block_partial(conv_filters[-2], conv_filters[-1])

        # Up path
        # Now we'll do the same we did on the down path, but mirrored. We also
        # need to account for the skip connections, that's why we sum the
        # channels for both outputs. That basically means that we are
        # concatenating with the skip connection, and not suming.
        self.up = nn.ModuleList([
            block_partial(f_in, f_out) for f_in, f_out in zip(
                deconv_in, deconv_out
            )
        ])

    def encode(self, input_s):
        # We need to keep track of the convolutional outputs, for the skip
        # connections.
        down_inputs = []
        for c in self.down:
            c.to(self.device)
            input_s = F.dropout3d(
                c(input_s), self.dropout, self.training
            )
            down_inputs.append(input_s)
            # Remember that pooling is optional
            if self.pooling:
                input_s = F.max_pool3d(input_s, 2)

        self.u.to(self.device)
        input_s = F.dropout3d(self.u(input_s), self.dropout, self.training)

        return down_inputs, input_s

    def decode(self, input_s, down_inputs):
        for d, i in zip(self.up, down_inputs[::-1]):
            d.to(self.device)
            # Remember that pooling is optional
            if self.pooling:
                input_s = F.dropout3d(
                    d(
                        torch.cat(
                            (F.interpolate(input_s, size=i.size()[2:]), i),
                            dim=1
                        )
                    ),
                    self.dropout,
                    self.training
                )
            else:
                input_s = F.dropout3d(
                    d(torch.cat((input_s, i), dim=1)),
                    self.dropout,
                    self.training
                )

        return input_s

    def forward(self, input_s, keepfeat=False):
        down_inputs, input_s = self.encode(input_s)

        features = down_inputs + [input_s] if keepfeat else []

        input_s = self.decode(input_s, down_inputs)

        output = (input_s, features) if keepfeat else input_s

        return output

    def compute_filters(self):
        conv_in = [self.n_inputs] + self.filters[:-2]
        conv_out = self.filters[:-1]
        down_out = self.filters[-2::-1]
        up_out = self.filters[:0:-1]
        deconv_in = list(map(sum, zip(down_out, up_out)))
        deconv_out = down_out
        return conv_in, conv_out, deconv_in, deconv_out


class BaseConv3dBlock(BaseModel):
    def __init__(self, filters_in, filters_out, kernel, inv):
        super().__init__()
        if not inv:
            self.conv = partial(
                nn.Conv3d, kernel_size=kernel, padding=kernel // 2
            )
        else:
            self.conv = partial(
                nn.ConvTranspose3d, kernel_size=kernel, padding=kernel // 2
            )

    def forward(self, inputs):
        return self.conv(inputs)

    @staticmethod
    def default_activation(n_filters):
        return nn.ReLU()


class Conv3dBlock(BaseConv3dBlock):
    def __init__(
            self, filters_in, filters_out, n_conv=1,
            kernel=3, norm=None, activation=None, inv=False
    ):
        super().__init__(filters_in, filters_out, kernel, inv)
        if activation is None:
            activation = self.default_activation
        self.block = nn.Sequential(
            self.conv(filters_in, filters_out),
            activation(filters_out),
            norm(filters_out)
        )
        self.extra = nn.ModuleList([
            nn.Sequential(
                self.conv(filters_out, filters_out),
                activation(filters_out),
                norm(filters_out)
            )
            for _ in range(n_conv - 1)
        ])

    def forward(self, inputs):
        out = self.block(inputs)
        for c in self.extra:
            out = c(out)
        return out


class ResConv3dBlock(BaseConv3dBlock):
    def __init__(
            self, filters_in, filters_out, n_conv=1,
            kernel=3, norm=None, activation=None, inv=False
    ):
        super().__init__(filters_in, filters_out, kernel, inv)
        if activation is None:
            activation = self.default_activation
        if n_conv < 1:
            n_conv = 1
        if not inv:
            conv = nn.Conv3d
        else:
            conv = nn.ConvTranspose3d

        self.first = nn.Sequential(
            self.conv(filters_in, filters_out),
            activation(filters_out),
            norm(filters_out)
        )

        if filters_in != filters_out:
            self.res = conv(filters_in, filters_out, 1)
        else:
            self.res = None

        self.seq = nn.ModuleList([
            nn.Sequential(
                self.conv(filters_out, filters_out),
                activation(filters_out),
                norm(filters_out)
            )
            for _ in range(n_conv - 1)
        ])

    def forward(self, inputs):
        res = inputs if self.res is None else self.res(inputs)
        out = self.first(inputs) + res
        for c in self.seq:
            out = c(out) + res
        return out


class SimpleUNet(BaseModel):
    def __init__(
            self,
            conv_filters=None,
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
            n_images=3,
            base_loss='xent',
            lr=1e-5,
            dropout=0,
            verbose=0,
    ):
        super().__init__()
        # We are applying the sigmoid function as part of the loss function
        # for a single reason. We want to define a loss functions whose
        # derivative with respect to the output of the network is defined
        # to follow certain properties. As such, we want the sigmoid to be
        # part of the derivative to avoid the derivative of the sigmoid and
        # its issues when not taking into account inside the loss.
        losses = {
            'xent': lambda x, y: F.binary_cross_entropy(
                torch.sigmoid(x), y.type_as(x).to(x.device)
            ),
            'xent_w': lambda x, y: focal_loss(
                torch.sigmoid(x), y.type_as(x).to(x.device), alpha=0.75, gamma=0
            ),
            'gdsc': partial(gendsc_loss, batch=False),
            'gdsc_b': gendsc_loss,
            'dsc': partial(gendsc_loss, w_bg=0, w_fg=1),
            'focal': lambda x, y: focal_loss(
                torch.sigmoid(x), y.type_as(x).to(x.device), alpha=0
            ),
            'focal_w1': lambda x, y: focal_loss(
                torch.sigmoid(x), y.type_as(x).to(x.device), alpha=0.25
            ),
            'focal_w2': lambda x, y: focal_loss(
                torch.sigmoid(x), y.type_as(x).to(x.device), alpha=0.75
            ),
            'new': lambda x, y: new_loss(
                x, y.type_as(x).to(x.device)
            ),
        }
        self.init = False
        # Init values
        if conv_filters is None:
            # self.conv_filters = list([32, 64, 256, 1024])
            self.conv_filters = list([32, 64, 128, 256])
        else:
            self.conv_filters = conv_filters
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.device = device
        self.dropout = dropout

        # <Parameter setup>

        self.ae = Autoencoder(
            self.conv_filters, device, n_images,
            # block=partial(ResConv3dBlock, n_conv=3),
            block=Conv3dBlock,
            pooling=True, norm=norm_f
        )
        self.ae.to(device)
        self.segmenter = nn.Conv3d(self.conv_filters[0], 1, 1)
        self.segmenter.to(device)

        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'loss',
                'weight': 1,
                'f': lambda p, t: losses[base_loss](
                    torch.sigmoid(p), t
                )
            },
        ]

        self.val_functions = [
            {
                'name': 'dsc',
                'weight': 1,
                'f': lambda p, t: dsc_binary_loss(
                    torch.sigmoid(p), t
                )
            },
            {
                'name': 'fn',
                'weight': 0,
                'f': lambda p, t: tp_binary_loss(
                    torch.sigmoid(p), t
                )
            },
            {
                'name': 'fp',
                'weight': 0,
                'f': lambda p, t: tn_binary_loss(
                    torch.sigmoid(p), t
                )
            },
            {
                'name': 'gdsc',
                'weight': 0,
                'f': lambda p, t: gendsc_loss(
                    torch.sigmoid(p), t
                )
            },
            {
                'name': 'xent',
                'weight': 0,
                'f': lambda p, t: F.binary_cross_entropy(
                    torch.sigmoid(p), t.type_as(p).to(p.device)
                )
            },
        ]

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.SGD(model_params, lr=lr)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def forward(self, data):
        features = self.ae(data)
        seg = self.segmenter(features)
        return seg

    def lesions(self, data):
        # Init
        self.eval()

        with torch.no_grad():
            data_tensor = to_torch_var(
                np.expand_dims(data, axis=0), self.device
            )
            features = self.ae(data_tensor)
            seg_tensor = torch.sigmoid(self.segmenter(features))
            torch.cuda.empty_cache()

            seg = seg_tensor[0, 0].cpu().numpy()

            return seg

    def lesions_patch(
            self, data, patch_size, batch_size,
            verbose=2
    ):
        # Init
        self.eval()

        # Init
        t_in = time.time()

        # This branch is only used when images are too big. In this case
        # they are split in patches and each patch is trained separately.
        # Currently, the image is partitioned in blocks with no overlap,
        # however, it might be a good idea to sample all possible patches,
        # test them, and average the results. I know both approaches
        # produce unwanted artifacts, so I don't know.
        # Initial results. Filled to 0.
        seg = np.zeros(data.shape[1:])
        counts = np.zeros(data.shape[1:])

        # The following lines are just a complicated way of finding all
        # the possible combinations of patch indices.
        steps = [
            list(
                range(0, lim - patch_size, patch_size // 4)
            ) + [lim - patch_size]
            for lim in data.shape[1:]
        ]

        steps_product = list(itertools.product(*steps))
        batches = range(0, len(steps_product), batch_size)
        n_batches = len(batches)

        # The following code is just a normal test loop with all the
        # previously computed patches.
        for bi, batch in enumerate(batches):
            # Here we just take the current patch defined by its slice
            # in the x and y axes. Then we convert it into a torch
            # tensor for testing.
            slices = [
                (
                    slice(xi, xi + patch_size),
                    slice(xj, xj + patch_size),
                    slice(xk, xk + patch_size)
                )
                for xi, xj, xk in steps_product[batch:(batch + batch_size)]
            ]

            data_batch = [
                data[slice(None), xslice, yslice, zslice]
                for xslice, yslice, zslice in slices
            ]

            # Testing itself.
            with torch.no_grad():
                data_tensor = to_torch_var(np.stack(data_batch, axis=0))

                seg_out = self(data_tensor)
                torch.cuda.empty_cache()

            # Then we just fill the results image.
            for si, (xslice, yslice, zslice) in enumerate(slices):
                counts[xslice, yslice, zslice] += 1
                seg_bi = seg_out[si, 0].cpu().numpy()
                seg[xslice, yslice, zslice] += seg_bi

            # Printing
            if verbose > 0:
                self.print_batch(bi, n_batches, 0, 1, t_in, t_in)

        if verbose > 0:
            print('\033[K', end='\r')

        seg /= counts
        return seg

    def print_batch(self, patch_j, n_patches, i, n_cases, t_in, t_case_in):
        init_c = '\033[38;5;238m'
        percent = 20 * (patch_j + 1) // n_patches
        progress_s = ''.join(['-'] * percent)
        remainder_s = ''.join([' '] * (20 - percent))

        t_out = time.time() - t_in
        t_case_out = time.time() - t_case_in
        time_s = time_to_string(t_out)

        t_eta = (t_case_out / (patch_j + 1)) * (n_patches - (patch_j + 1))
        eta_s = time_to_string(t_eta)
        pre_s = '{:}Case {:03}/{:03} ({:03d}/{:03d}) [{:}>{:}]' \
                ' {:} ETA: {:}'
        batch_s = pre_s.format(
            init_c, i + 1, n_cases, patch_j + 1, n_patches,
            progress_s, remainder_s, time_s, eta_s + '\033[0m'
        )
        print('\033[K', end='', flush=True)
        print(batch_s, end='\r', flush=True)

