
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import data_loader

import os
from torchvision.utils import save_image
import torch.autograd as autograd

import torchvision

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sample_noise(batch_size, dim):
    """
    Given the inputs batch_size, and dim return a tensor of values between (L, U)
    :param batch_size (int): size of the batch
    :param dim (int): size of the vector of dimensions
    :return: tensor of a random values between (L, U)

    Used to generate images in GANs
    """

    #TODO take the L and U as inputs of the function
    L = -1 #lower bound
    U = 1 #upper bound

    noise = (L - U) * torch.rand((batch_size, dim)) + U

    return noise

class Flatten(nn.Module):
    """
    Given a tensor of Batch * Color * Height * Width, flatten it and make it 1D.
    Used for Linear GANs

    Usable in nn.Sequential
    """
    def forward(self, x):

        B, C, H, W = x.size()

        return x.view(B, -1) #returns a vector that is B * (C * H * W)

class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (Batch, C*H*W) and reshapes it
    to produce an output of shape (Batch, C, H, W).

    C = Color Channels
    H = Heigh
    W = Width
    """

    def __init__(self, B=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()

        self.B = B
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        return x.view(self.B, self.C, self.H, self.W)


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.

    As per https://github.com/pytorch/pytorch/issues/751

    :param input: PyTorch Tensor of shape (N, )
    :param target: PyTorch Tensor of shape (N, ). An indicator variable that is 0 or 1
    :return:
    """

    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()

    return loss.mean()


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss for Vanilla GANs

    :param logits_real: PyTorch Tensor of shape(N, ). Gives scores for the real data
    :param logits_fake: PyTorch Tensor of shape(N, ). Gives scores for the fake data
    :return: PyTorch Tensor containing the loss for the discriminator
    """

    labels = torch.ones(logits_real.size()).to(device) #label used to indicate whether it's real or not

    loss_real = nn.MSELoss()(logits_real, labels) #real data
    loss_fake = nn.MSELoss()(logits_fake, 1 - labels) #fake data

    loss = loss_real + loss_fake

    return loss.to(device) * 1/2

def generator_loss(logits_fake):
    """
    Computes the generator loss

    :param logits_fake: PyTorch Tensor of shape (N, ). Gives scores for the real data
    :return: PyTorch tensor containing the loss for the generator
    """

    labels = torch.ones(logits_fake.size()).to(device)

    loss = nn.MSELoss()(logits_fake, labels)

    return loss.to(device)


def LS_discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss for Vanilla GANs

    :param logits_real: PyTorch Tensor of shape(N, ). Gives scores for the real data
    :param logits_fake: PyTorch Tensor of shape(N, ). Gives scores for the fake data
    :return: PyTorch Tensor containing the loss for the discriminator
    """

    labels = torch.ones(logits_real.size()).to(device) #label used to indicate whether it's real or not

    #A, B, C values based on https://arxiv.org/pdf/1611.04076v3.pdf Page 8
    loss_real = 1/2 * ((logits_real - labels) ** 2).mean() #real data
    loss_fake = 1/2 * ((logits_fake) ** 2).mean() #fake data

    loss = loss_real + loss_fake

    return loss.to(device)

def LS_generator_loss(logits_fake):
    """
    Computes the generator loss

    :param logits_fake: PyTorch Tensor of shape (N, ). Gives scores for the real data
    :return: PyTorch tensor containing the loss for the generator
    """

    labels = torch.ones(logits_fake.size()).to(device)

    loss = 1/2 * ((logits_fake - labels) ** 2).mean()

    return loss.to(device)

def get_optimizer(model, lr=0.0005):
    """
    Takes in PyTorch model and returns the Adam optimizer associated with it
    :param model:
    :param lr:
    :return:
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    return optimizer

def show_images(images, filename, iterations, title=None):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')

        '''global title'''
        if title == None:
            title = 'StarGan After %s iterations'
        plt.suptitle(title %iterations)
        plt.imshow(img.reshape([sqrtimg, sqrtimg]))
        plt.savefig(filename)

def one_hot_encoder(index_array, n_classes = 10):
    '''
    One hot encoder that takes in an array of labels (ex.  array([1, 0, 3])) and returns
    a n-dimensional array that is one hot encoded
    :param index_array (array of ints): an array that holds class labels
    :param n_classes (int): number of classes
    :return:
    '''

    batch_size = index_array.shape[0]

    b = np.zeros((batch_size, n_classes))
    b[np.arange(batch_size), index_array] = 1
    return np.float32(b)

def categorical_label_generator(batch_size=128, n_classes=10):
    '''
    Function that returns an array with a label between [0, 10] for an element with size batch_size
    :param batch_size (int): length of an array
    :param n_classes (int): total number of classes
    :return:
    '''

    array = np.random.choice(n_classes, batch_size)

    return array

def generate_fake_label(batch_size=128, n_classes=10, specific=None):
    '''

    :param batch_size (int): Batch size
    :param n_classes (int): Number of classes
    :param specific (int): specific class label we want
    :return:
    '''

    if specific == None:
        fake_labels = categorical_label_generator(batch_size, n_classes=n_classes)
    else:
        fake_labels = np.ones(batch_size) * specific
        fake_labels = fake_labels.astype(int)

    fake_labels = one_hot_encoder(fake_labels, n_classes)

    return fake_labels

def spatially_replicate(z_code, output_shape):
    '''
    Takes in a hidden Batch X Z code and returns a tensor of size Batch X H X W X Z by replicating
    :param z_code (tensor): hidden code
    :return: Batch X H X W X Z sized tensor
    '''

    output = z_code.repeat(output_shape, output_shape)

    return output.view(z_code.shape[0], z_code.shape[1], output_shape, output_shape)

def randn_z_dimension(output_shape):
    '''
    Takes in an output_shape and returns a tensor of size Batch x H x W x Z
    :param output_shape:
    :return:
    '''

    output = torch.randn(output_shape)

    return output

def expand_spatially(input, image_shape):

    label = input.view((input.shape[0], input.shape[1], 1, 1))
    label = label.repeat(1, 1, image_shape, image_shape)

    return label


def save_images_to_directory(image_tensor, directory, filename, nrow=8):
    directory = directory
    image = image_tensor.cpu().data

    save_name = os.path.join('%s' % directory, '%s' % filename)
    save_image(image, save_name, nrow=nrow)

    return filename

def calc_gradient_penalty(netD, real_data, fake_data):
    '''https://discuss.pytorch.org/t/how-to-implement-gradient-penalty-in-pytorch/1656/12'''
    '''https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py'''
    #print real_data.size()
    alpha = torch.Tensor(np.random.random((real_data.size(0), 1, 1, 1))).to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data).to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates, _ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = gradients.view(gradients.size(0), -1)
    gradient_penalty = torch.sqrt(torch.sum(gradient_penalty ** 2, dim=1))

    return torch.mean((gradient_penalty - 1) ** 2)


def plotter(env_name, num_episodes, rewards_list, ylim):
    '''
    Used to plot the average over time
    :param env_name:
    :param num_episodes:
    :param rewards_list:
    :param ylim:
    :return:
    '''
    x = np.arange(0, num_episodes)
    y = np.asarray(rewards_list)
    plt.plot(x, y)
    plt.ylim(top=ylim + 10)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Avg Rewards Last 100 Episodes")
    plt.title("Rewards Over Time For %s" %env_name)
    plt.savefig("progress.png")
    plt.close()

def raw_score_plotter(scores):
    '''
    used to plot the raw score
    :param scores:
    :return:
    '''
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Train Loss')
    plt.xlabel('Number of Iterations')
    plt.title("Loss Over Time")
    plt.savefig("Train_Loss.png")
    plt.close()

def linear_LR(epoch, opts):

    if epoch < opts.const_epoch:
        lr = opts.lr
    else:
        lr = np.linspace(opts.lr, 0, (opts.adaptive_epoch + 1))[epoch - opts.const_epoch]

    return lr

def stargan_side_by_side_images(opts, generator, original_image, original_label=None, attribute=data_loader.attribute_combinations()):

    col_l_lim = 0
    label_given = True

    if original_label == None:
        label_given = False
        original_label = torch.zeros(opts.num_classes).to(device)

    with torch.no_grad():
        image = original_image.clone()
        for sub_attribute in attribute:
            for j in range(len(sub_attribute)):
                right_lim = len(sub_attribute[j])
                one_hot_label = sub_attribute[j]  # one_hot_encoder := binary attributes in sub_attributes
                new_label = original_label.clone()
                if label_given:
                    new_label[:, col_l_lim: col_l_lim + len(sub_attribute[j])] = torch.from_numpy(one_hot_label).float().to(device)  # change the label that only correspond to sub-attribute being changed.
                else:
                    new_label[col_l_lim: col_l_lim + len(sub_attribute[j])] = torch.from_numpy(one_hot_label).float().to(device)
                    new_label.unsqueeze_(0)
                new_label_ = new_label.float().to(device)
                fake_image = generator(original_image, new_label_).detach()
                image = torch.cat([image, fake_image], -1)  # column wise concat

            col_l_lim += right_lim

    return image


def save_grid_images(image_tensor, directory, filename, rows=3):

    grid_img = torchvision.utils.make_grid(image_tensor, nrow=rows)

    filename = save_images_to_directory(grid_img, directory, filename)

    return filename
