import torch
import data_loader
import image_folder

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_dir = '/home/youngwook/Documents/celeb/img_align_celeba/'
attribute_file = '/home/youngwook/Documents/celeb/list_attr_celeba.txt'
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

data = data_loader.read_data(opts, folder_names[0])

train_x, test_x = data.train_test_split()

train_data = data_loader.CelebA_DataLoader(train_x, folder_names[0], transform=transform, attribute_file=attribute_file, size=opts.resize, randomcrop=opts.image_shape)
trainloader = DataLoader(train_data, batch_size=opts.batch, shuffle=True, num_workers=4)

test_data = data_loader.CelebA_DataLoader(test_x, folder_names[0], transform=transform, attribute_file=attribute_file, size=opts.resize, randomcrop=opts.image_shape)
testloader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=8)

from network import resnet50
from train import GAN_Trainer
from test import Gan_Tester

'''Discriminator'''
D = resnet50().to(device)

'''Optimizers'''
import torch.optim as optim

D_optim = optim.Adam(D.parameters(), lr=opts.lr, betas=(opts.beta1, opts.beta2))

'''run training'''
trainer = GAN_Trainer(opts, D, D_optim, trainloader)
trainer.train()

tester = Gan_Tester(opts, D, testloader, checkpoint='./model/checkpoint_-1.pth')
tester.test()
