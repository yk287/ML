
import matplotlib.pyplot as plt
import numpy as np
import torch


import matplotlib.ticker as ticker

def confusion_plot(matrix, y_category):
    '''
    A function that plots a confusion matrix
    :param matrix: Confusion matrix
    :param y_category: Names of categories.
    :return: NA
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + y_category, rotation=90)
    ax.set_yticklabels([''] + y_category)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()


def accuracy(net, loader):
    '''
    A function that returns total number of correct predictions and total comparisons
    given a neural net and a pytorch data loader
    :param net: neural net
    :param loader: data loader
    :return:
    '''

    correct = 0
    total = 0

    for images, labels in iter(loader):

        output = net.forward(images)

        _, prediction = torch.max(output.data, 1)

        total += labels.shape[0] #accumulate by batch_size
        correct += (prediction == labels).sum() #accumulate by total_correct

    return correct, total

def prediction_accuracy(net, images, labels):
    '''
    A function that returns total number of correct predictions and total comparisons
    given a neural net and a pytorch data loader
    :param net: neural net
    :param loader: data loader
    :return:
    '''

    output = net.forward(images)

    _, prediction = torch.max(output.data, 1)

    total = labels.shape[0] #accumulate by batch_size
    correct = (prediction == labels).sum() #accumulate by total_correct

    return correct, total


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

def confusion_plot(matrix, y_category):
    '''
    A function that plots a confusion matrix
    :param matrix: Confusion matrix
    :param y_category: Names of categories.
    :return: NA
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + y_category, rotation=90)
    ax.set_yticklabels([''] + y_category)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()

def linear_LR(epoch, opts):

    if epoch < opts.const_epoch:
        lr = opts.lr
    else:
        lr = np.linspace(opts.lr, 0, (opts.adaptive_epoch + 1))[epoch - opts.const_epoch]

    return lr

