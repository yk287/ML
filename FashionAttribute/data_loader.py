#Inspired by https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/data/colorization_dataset.py

import numpy as np
import random

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
from image_folder import get_images

import torchvision.transforms.functional as TVF

from collections import defaultdict

#for reading .mat file
import scipy.io
mat = scipy.io.loadmat('/home/youngwook/Documents/ClothingAttributeDataset/labels/category_GT.mat')
test = mat['GT']
from sklearn import preprocessing


class read_data():
    """
    Dataloader class used to load in data in an image folder.
    Made it so that it performs a fixed set of transformations to a pair of images in different folders
    """

    def __init__(self, opts, img_folder, attribute_file):
        '''

        :param opts:
        :param img_folder:
        :param attribute_file:
        '''

        self.img_folder = img_folder
        self.attribute_file = attribute_file
        self.opts = opts

        self.file_names = get_images(img_folder)
        #load_attributes
        self.load_attribute()
        self.data_dictionary()

    def data_dictionary(self):

        self.dataIdx = {}

        for name in self.file_names:
            self.dataIdx[name] = len(self.dataIdx)

    def load_attribute(self):
        """
        loads attributes from self.attribute_file using the column as a key to a dictionary and rest of the columns
        as values.
        """

        attributes = scipy.io.loadmat(self.attribute_file)
        #just get the attriutes
        attributes = attributes['GT']
        #labels contain NAN I just converted it to 0
        self.attributes = np.nan_to_num(attributes)

    def train_test_split(self):
        '''
        Given the data that's loaded above, returns a data that's been split into training and test set.

        :param split: what % of the data should be allocated to the training dataset
        :return: training, test which are list of tuples (Label, Name)
        '''

        p = np.random.permutation(len(self.file_names))

        data = np.asarray(self.file_names)[p]
        tags = np.asarray(self.attributes)[p]

        cutoff = np.int(len(data) * self.opts.split)

        train_x, train_y, test_x, test_y  = data[:cutoff], tags[:cutoff], data[cutoff:], tags[cutoff:]

        return train_x, train_y, test_x, test_y

class CelebA_DataLoader(Dataset):
    """
    Dataloader class used to load in data in an image folder.
    Made it so that it performs a fixed set of transformations to a pair of images in different folders
    """

    def __init__(self, opts, img_file, attribute_file, transform=None, size =256, randomcrop = 224, hflip=0.5, vflip=0.5, train=True):
        '''
        :param img_folder:
        :param transform:
        :param additional_transform:
        :param final_transformation:
        '''

        self.file_names = img_file
        self.attribute_file = attribute_file

        self.opts = opts

        self.size = size #for resizing
        self.randomCrop = randomcrop #randomcrop
        self.hflip = hflip
        self.vflip = vflip
        self.transforms = transform
        self.train = train

    def __len__(self):

        return len(self.file_names)

    def __getitem__(self, index):

        left_image = Image.open(self.file_names[index]).convert('RGB')
        attributes = self.attribute_file[index]

        '''
        Resize
        '''
        resize = transforms.Resize(size=(self.size, self.size))
        left_image = resize(left_image)

        '''
        RandomCrop
        '''
        if self.train:
            i, j, h, w = transforms.RandomCrop.get_params(
                left_image, output_size=(self.randomCrop, self.randomCrop)
            )
            left_image = TVF.crop(left_image, i, j, h, w)

            if random.random() >= self.hflip:
                left_image = TVF.hflip(left_image)

        left_image = self.transforms(left_image)

        return left_image, attributes
