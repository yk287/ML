
import util
import numpy as np
import torch
from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class trainer():

    def __init__(self, opts, trainloader, model, optimizer, criterion):

        self.opts = opts
        self.trainloader = trainloader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self):
        steps = 0

        loss_deque = deque(maxlen=100)
        train_loss = []

        for e in range(self.opts.epoch):
            running_loss = 0

            correct = 0
            total = 0

            for images, labels in iter(self.trainloader):
                steps += 1

                output = self.model(images.to(device))

                self.optimizer.zero_grad()
                loss = self.criterion(output, labels.to(device))
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                pred = torch.max(output, 1)[1]

                for pred, label in zip(pred, labels):
                    if pred.cpu().item() == label.item():
                        correct += 1
                    total += 1

                loss_deque.append(loss.cpu().item())
                train_loss.append(np.mean(loss_deque))

                if steps % self.opts.print_every == 0:
                    print("Epoch: {}/{}...".format(e + 1, self.opts.epoch),
                          "LossL {:.4f}".format(running_loss / self.opts.print_every),
                          "Running Accuracy {:4f}".format(correct / np.float(total)))

                    running_loss = 0

        util.raw_score_plotter(train_loss)