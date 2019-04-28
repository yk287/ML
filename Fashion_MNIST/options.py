import argparse

class options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics

        # Training Options
        self.parser.add_argument('--epoch', type=int, nargs='?', default=30, help='total number of training episodes')
        self.parser.add_argument('--show_every', type=int, nargs='?', default=500, help='How often to show images')
        self.parser.add_argument('--print_every', type=int, nargs='?', default=250, help='How often to print scores')
        self.parser.add_argument('--cpu_count', type=int, nargs='?', default=8, help='number of cpu used for dataloading')

        self.parser.add_argument('--print_model', type=bool, nargs='?', default=True, help='Prints the model being used')
        self.parser.add_argument('--batch', type=int, nargs='?', default=32, help='batch size to be used')

        self.parser.add_argument('--lr', type=int, nargs='?', default=0.0003, help='learning rate')
        self.parser.add_argument('--beta1', type=int, nargs='?', default=0.5, help='beta1 for adam optimizer')
        self.parser.add_argument('--beta2', type=int, nargs='?', default=0.999, help='beta2 for adam optimizer')

        self.parser.add_argument('--lrelu_val', type=int, nargs='?', default=0.20, help='leaky Relu Value')
        self.parser.add_argument('--dropout', type=float, nargs='?', default=0.30, help='dropout rate')

        self.parser.add_argument('--resume', type=bool, nargs='?', default=False, help='Resume Training')
        self.parser.add_argument('--save_progress', type=bool, nargs='?', default=True, help='save training progress')

        self.parser.add_argument('--num_classes', type=int, nargs='?', default=10, help='number of classes')

        # Image
        self.parser.add_argument('--resize', type=int, nargs='?', default=144, help='Image Resize size')
        self.parser.add_argument('--num_images', type=int, nargs='?', default=2, help='number of images in a single image file')
        self.parser.add_argument('--image_shape', type=int, nargs='?', default=128, help='height of a square image')
        self.parser.add_argument('--channel_in', type=int, nargs='?', default=1, help='number of input channels')
        self.parser.add_argument('--channel_out', type=int, nargs='?', default=1, help='number of output channels')
        self.parser.add_argument('--channel_up', type=int, nargs='?', default=8, help='initial channel increasing')

        #Discriminator Options
        self.parser.add_argument('--n_discrim_down', type=int, nargs='?', default=3, help='hidden layer configuration in a list form for D')
        self.parser.add_argument('--D_activation', type=str, nargs='?', default='lrelu', help='Activation function for the discriminator')
        self.parser.add_argument('--D_input_channel', type=int, nargs='?', default=1, help='size of input for the discriminator')
        self.parser.add_argument('--D_output_size', type=int, nargs='?', default=1, help='size of output for the discriminator')
        self.parser.add_argument('--D_channel_up', type=int, nargs='?', default=32,
                                 help='size of input for the discriminator')
        #Generator Options
        self.parser.add_argument('--use_bias', type=bool, nargs='?', default=True, help='whether to use bias or not')
        self.parser.add_argument('--n_conv_down', type=int, nargs='?', default=2, help='number of conv downs')
        self.parser.add_argument('--n_resblock', type=int, nargs='?', default=6, help='number of resnet')
        self.parser.add_argument('--n_conv_up', type=int, nargs='?', default=2, help='number of conv up')


        self.parser.add_argument('--G_hidden', type=int, nargs='+', default=[1024, 1024], help='hidden layer configuration in a list form for G')
        self.parser.add_argument('--G_activation', type=str, nargs='?', default='relu', help='Activation function for the generator')
        self.parser.add_argument('--noise_dim', type=int, nargs='?', default=96, help='size of noise input for the generator')
        self.parser.add_argument('--G_output_size', type=int, nargs='?', default=784, help='size of output for the discriminator')
        self.parser.add_argument('--G_out_activation', type=str, nargs='?', default='tanh', help='final output activator')

    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()

        return self.opt



"""
options = options()
opts = options.parse()
batch = opts.batch
"""