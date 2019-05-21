#Inspired by https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/data/colorization_dataset.py

import numpy as np
import random

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
from image_folder import get_images

import torchvision.transforms.functional as TVF

from sklearn import preprocessing

attributes = [
    'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'
]

hair_color_attributes = [
    'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'
]

lipstick_attribute = [
    'Wearing_Lipstick'
]

bang = [
    'Bangs'
]

facial = [
    '5_o_Clock_Shadow'
]

target_attributes = [hair_color_attributes[:]]
target_attributes1 = [hair_color_attributes[:]]

#target_attributes = [facial[:], hair_color_attributes[:], bang[:], lipstick_attribute[:]]
#target_attributes1 = [facial[:], hair_color_attributes[:], bang[:], lipstick_attribute[:]]

def attribute_combinations(attributes=target_attributes1):

    binary_attributes = []
    for attr in attributes:
        lb = preprocessing.LabelBinarizer()
        if len(attr) == 1:
            '''in case the list only has 1 value, like bang, append an empty str to indicate "off" '''
            '''otherwise the binarizer only outputs 1 value "1"'''
            psudo_attribute = ['']
            attr += psudo_attribute
        binary_attributes.append(lb.fit_transform(attr))

    return binary_attributes


class CelebA_DataLoader(Dataset):
    """
    Dataloader class used to load in data in an image folder.
    Made it so that it performs a fixed set of transformations to a pair of images in different folders
    """

    def __init__(self, file_list, img_folder, attribute_file, target_attributes=attributes,  target_attributes_list=target_attributes, transform=None, size =256, randomcrop = 224, hflip=0.5, vflip=0.5, train=True):
        '''
        :param img_folder:
        :param transform:
        :param additional_transform:
        :param final_transformation:
        '''

        self.attribute_file = attribute_file
        self.attributes = {}

        self.target_attributes=target_attributes
        self.target_attributes_list = target_attributes_list

        self.size = size #for resizing
        self.randomCrop = randomcrop #randomcrop
        self.hflip = hflip
        self.vflip = vflip
        self.transforms = transform
        self.train = train

        self.file_names = file_list
        self.img_folder = img_folder
        #load_attributes
        self.load_attribute()

    def load_attribute(self):
        """
        loads attributes from self.attribute_file using the column as a key to a dictionary and rest of the columns
        as values.
        """

        lines = [line.rstrip() for line in open(self.attribute_file, 'r')]
        self.all_attr_names = lines[0].split(',')
        #print(self.all_attr_names)
#        self.idx = []
        #get the index of target attributes
#        for attr in self.all_attr_names:
#            self.idx.append(self.all_attr_names.index(attr))
#        self.target_attr_names = [self.all_attr_names[idx] for idx in self.idx]

        for i in range(1,len(lines)):
            line_split = lines[i].split(',')
            attributes = line_split[1:-1].copy()
            self.attributes[line_split[0]] = attributes

    def attribute_chooser(self, attribute_list):

        binary = np.zeros(len(attribute_list), dtype=float)
        if len(attribute_list) > 1:
            choice = np.random.randint(0, len(attribute_list))
        else:
            if np.random.rand() <= 0.5:
                choice = 0
            else:
                return attribute_list[0], binary

        chosen = attribute_list[choice]
        chosen_idx = attribute_list.index(chosen)

        binary[chosen_idx] = 1

        return chosen, binary

    def get_transformed_attributes(self, image_name):

        target_attributes = self.attributes[image_name][:]

        for target_set in self.target_attributes_list:
            chosen_attribute, binary = self.attribute_chooser(target_set)
            for idx, attr in enumerate(target_set):
                index = self.target_attributes.index(attr)
                target_attributes[index] = binary[idx]

        return target_attributes

    def list_to_tensor(self, input):
        input = list(map(float, input))
        input = np.asarray(input)

        return input

    def __len__(self):

        return len(self.file_names)

    def __getitem__(self, index):

        left_image = Image.open(self.file_names[index]).convert('RGB')
        image_name = self.file_names[index].replace(self.img_folder, "")

        orig_attributes = self.attributes[image_name]

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

        orig_label = (self.list_to_tensor(orig_attributes) + 1.0) / 2.0

        return left_image, orig_label

class read_data():
    """
    Dataloader class used to load in data in an image folder.
    Made it so that it performs a fixed set of transformations to a pair of images in different folders
    """

    def __init__(self, opts, img_folder, ):
        '''

        :param opts:
        :param img_folder:
        :param attribute_file:
        '''

        self.img_folder = img_folder
        self.opts = opts

        self.file_names = get_images(img_folder)

    def train_test_split(self):
        '''
        Given the data that's loaded above, returns a data that's been split into training and test set.

        :param split: what % of the data should be allocated to the training dataset
        :return: training, test which are list of tuples (Label, Name)
        '''

        p = np.random.RandomState(seed=self.opts.seeds).permutation(len(self.file_names))

        data = np.asarray(self.file_names)[p]

        cutoff = np.int(len(data) * self.opts.split)

        train_x, test_x = data[:cutoff], data[cutoff:]


        return train_x, test_x
