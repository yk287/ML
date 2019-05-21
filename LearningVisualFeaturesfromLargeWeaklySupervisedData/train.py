
import torch
import os
import torch.nn as nn
import numpy as np

import util
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from image_to_gif import image_to_gif
import torch.nn.functional as F

from collections import deque


class GAN_Trainer():
    def __init__(self, opts, D, D_solver, loader):

        self.opts = opts
        self.D = D
        self.D_solver = D_solver
        self.loader = loader
        self.checkpoint_path = './model/checkpoint_const1.pth'

    def save_progress(self, epoch, loss):

        directory = './model/'
        filename = 'checkpoint_%s.pth' % epoch

        path = os.path.join('%s' % directory, '%s' % filename)

        torch.save({
            'epoch': epoch,
            'loss': loss,
            'Discriminator_state_dict': self.D.state_dict(),
            'D_solver_state_dict': self.D_solver.state_dict(),
        }, path)

        print("Saving Training Progress")

    def load_progress(self,):

        # TODO get rid of hardcoding and come up with an automated way.

        checkpoint = torch.load(self.checkpoint_path)

        self.D.load_state_dict(checkpoint['Discriminator_state_dict'])

        self.D_solver.load_state_dict(checkpoint['D_solver_state_dict'])

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print("Training Loss From Last Session")
        return epoch, loss

    def train(self):
        """
        Vanilla GAN Trainer

        :param D: Discriminator
        :param G: Generator
        :param D_solver: Optimizer for D
        :param G_solver: Optimizer for G
        :param discriminator_loss:  Loss for D
        :param generator_loss:  Loss for G
        :param loader: Torch dataloader
        :param show_every: Show samples after every show_every iterations
        :param batch_size: Batch Size used for training
        :param noise_size: Dimension of the noise to use as input for G
        :param num_epochs: Number of epochs over the training dataset to use for training
        :return:
        """
        last_100_loss = deque(maxlen=100)
        last_100_g_loss =[]

        iter_count = 0

        last_epoch = 0
        if self.opts.resume:
            last_epoch, loss = self.load_progress()


        for epoch in range(self.opts.epoch- last_epoch):

            '''Adaptive LR Change'''
            for param_group in self.D_solver.param_groups:
                param_group['lr'] = util.linear_LR(epoch, self.opts)
                print('epoch: {}, D_LR: {:.4}'.format(epoch, param_group['lr']))

            if self.opts.save_progress:
                '''Save the progress before start adjusting the LR'''
                if epoch == self.opts.const_epoch:
                    self.save_progress(self.opts.const_epoch, np.mean(last_100_loss))

            for image, label in self.loader:

                '''Real Images'''
                image = image.to(device)

                '''one hot encode the real label'''

                label = label.float().to(device)
                '''Train Discriminator'''
                '''Get the logits'''

                real_logits_cls = self.D(image.to(device))

                loss = self.opts.cls_lambda * F.binary_cross_entropy_with_logits(real_logits_cls, label, reduction='sum') / real_logits_cls.size(0)


                self.D_solver.zero_grad()
                loss.backward()
                self.D_solver.step()  # One step Descent into loss

                '''Train Generator'''
                iter_count += 1

                last_100_loss.append(loss.cpu().item())
                last_100_g_loss.append(np.mean(last_100_loss))

                if iter_count % self.opts.print_every == 0:
                    print('Epoch: {}, Iter: {}, D: {:.4} '.format(epoch, iter_count, loss.item()))
                    util.raw_score_plotter(last_100_g_loss)

                if self.opts.save_progress:
                    if iter_count % self.opts.save_every == 0:
                        self.save_progress(epoch, np.mean(last_100_loss))

        if self.opts.save_progress:
            '''Save the progress before start adjusting the LR'''
            self.save_progress(-1, np.mean(last_100_loss))




