
import numpy as np
import torch
import confusionMatrix

import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class tester():

    def __init__(self, opts, trainloader, model):

        self.opts = opts
        self.trainloader = trainloader
        self.model = model

    def test(self):
        steps = 0

        correct = 0
        total = 0

        pred_list = []
        labels_list = []
        y_label = np.arange(self.opts.num_classes)
        y_label = [str(e) for e in y_label]

        self.model.eval()

        for images, labels in iter(self.trainloader):
            steps += 1
            # label to be fed into confusion matrix plot.

            output = self.model(images.to(device))

            pred = torch.max(output, 1)[1]

            pred_list.append(pred.cpu().item())
            labels_list.append(labels.item())

            if pred.cpu().item() == labels.item():
                correct += 1
            total += 1

        confusionMatrix.plot_confusion_matrix(np.array(labels_list, dtype=np.int), np.array(pred_list, dtype=np.int), np.array(y_label), title="ConfusionMatrix")
        plt.show()

        print("Test Accuracy", correct / float(total))

