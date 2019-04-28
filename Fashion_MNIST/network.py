
import torch
import torch.nn as nn
import torch.nn.functional as F


class discriminator(nn.Module):
    '''
    Patch Discriminator used in GANS.
    '''

    def __init__(self, opts):
        super(discriminator, self).__init__()

        self.opts = opts
        steps = []
        in_channel = opts.D_input_channel
        channel_up = opts.D_channel_up

        for i in range(opts.n_discrim_down):
            steps += [nn.Conv2d(in_channel, channel_up, 4, 2, 1), nn.BatchNorm2d(channel_up), nn.LeakyReLU(opts.lrelu_val, True), nn.Dropout3d(opts.dropout)]
            in_channel = channel_up
            channel_up *= 2

        cls = [nn.Conv2d(in_channel, opts.num_classes, 5, 1, 1)]

        self.model = nn.Sequential(*steps)
        self.cls = nn.Sequential(*cls)

    def forward(self, x):

        x = self.model(x)

        return self.cls(x).view((-1, self.opts.num_classes))