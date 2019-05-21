
import torch

#to get k nearest neighbors
from sklearn.neighbors import NearestNeighbors

import util
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from image_to_gif import image_to_gif



class Gan_Tester():
    def __init__(self, opts, G, loader, checkpoint):

        self.opts = opts
        self.G = G
        self.loader = loader
        self.checkpoint_path = checkpoint

    def load_progress(self,):

        checkpoint = torch.load(self.checkpoint_path)

        self.G.load_state_dict(checkpoint['Discriminator_state_dict'])

        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print("Training Loss From Last Session")
        return epoch, loss

    def test(self):

        _, _ = self.load_progress()

        directory = './img/test/'

        self.G.eval()

        image_list = []

        steps = 0
        for image, label in self.loader:

            '''Real Images'''

            features = self.G(image.to(device)).detach().cpu().numpy()

            nbrs = NearestNeighbors(n_neighbors=self.opts.num_neigh, algorithm='ball_tree').fit(features)

            distances, indices = nbrs.kneighbors(features)

            for idx, i in enumerate(indices):
                stacked_image = torch.cat([image[i]], 0)
                image_list.append(util.save_grid_images(stacked_image, directory, 'imges_%s_%s.png' %(steps, idx), rows=self.opts.num_neigh))


            if steps == 5:
                break

            steps += 1

        #create a gif
        image_to_gif(directory, image_list, duration=2, gifname='test_images')




