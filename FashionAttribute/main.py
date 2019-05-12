
import torch
import torch.nn as nn
import torch.optim as optim

import data_loader
import image_folder

#import test_loader

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_dir = '/home/youngwook/Documents/ClothingAttributeDataset/images'
attribute_file = '/home/youngwook/Documents/ClothingAttributeDataset/labels/category_GT.mat'
folder_names = image_folder.get_folders(image_dir)

#options
from options import options
options = options()
opts = options.parse()

#Download and load the training data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])

train_image, train_tag, test_image, test_tag = data_loader.read_data(opts, folder_names[0], attribute_file=attribute_file).train_test_split()


train_data = data_loader.CelebA_DataLoader(opts, train_image, train_tag, transform=transform, size=opts.resize, randomcrop=opts.image_shape)
trainloader = DataLoader(train_data, batch_size=opts.batch, shuffle=True, num_workers=4)

from network import discriminator
import trainer
from tester import tester

output_size = opts.num_classes
model = discriminator(opts).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))

#trains the model
trainer = trainer.trainer(opts, trainloader, model, optimizer, criterion)
trainer.train()

test_data = data_loader.CelebA_DataLoader(opts, test_image, test_tag,  transform=transform, size=opts.resize, randomcrop=opts.image_shape)
testloader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4)

#trains the model
tester = tester(opts, testloader, model)
tester.test()


