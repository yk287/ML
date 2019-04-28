
import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

#load mnist dataset and define network
from torchvision import datasets, transforms

from network import discriminator

from options import options
options = options()
opts = options.parse()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(([0.5]), ([0.5])),])

#Download and load the training data
trainset = datasets.FashionMNIST('FMNIST_data/', download=True, train= True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opts.batch, shuffle=True)

testset = datasets.FashionMNIST('FMNIST_data/', download=True, train=False, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

import trainer
from tester import tester

#Hyperparameters for our network

output_size = opts.num_classes
model = discriminator(opts).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))

#trains the model
trainer = trainer.trainer(opts, trainloader, model, optimizer, criterion)
trainer.train()

#trains the model
tester = tester(opts, testloader, model)
tester.test()

