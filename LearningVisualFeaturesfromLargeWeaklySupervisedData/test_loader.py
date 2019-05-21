
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
from image_folder import get_images

class CelebA_DataLoader(Dataset):
    """
    Dataloader class used to load in data in an image folder.
    Made it so that it performs a fixed set of transformations to a pair of images in different folders
    """

    def __init__(self, img_folder, transform=None, size =256, randomcrop = 224, hflip=0.5, vflip=0.5, train=True):
        '''
        :param img_folder:
        :param transform:
        :param additional_transform:
        :param final_transformation:
        '''

        self.img_folder = img_folder

        self.size = size #for resizing
        self.randomCrop = randomcrop #randomcrop
        self.hflip = hflip
        self.vflip = vflip
        self.transforms = transform
        self.train = train

        self.file_names = get_images(img_folder)

        #load_attributes

    def __len__(self):

        return len(self.file_names)

    def __getitem__(self, index):

        left_image = Image.open(self.file_names[index]).convert('RGB')

        '''
        Resize
        '''

        resize = transforms.Resize(size=(self.size, self.size))
        left_image = resize(left_image)

        '''
        RandomCrop
        '''

        left_image = self.transforms(left_image)

        return left_image

