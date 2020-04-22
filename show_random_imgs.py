from random import randint
import os
from matplotlib import pyplot as plt
import glob
import argparse
from skimage.io import sift, imread, imsave, imshow

def show_random_imgs(images, x, y, figsize = (10, 10)):
    n = x * y
    fig, axes = plt.subplots(x, y, figsize = figsize, sharex=True, sharey=True)
    image_paths = os.path.join(images, '*.jpg')
    im = glob.glob(image_paths)
    print('Number of images ' + str(len(im)))
    ax = axes.ravel()
    for i in range(n):
        k = randint(0,len(im)-1)
        image = imread(im[k])
        ax[i].imshow(image)
        #ax[i].set_title(os.path.basename(im[k]).split('_')[0], fontsize = 'xx-small')
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Shows a grid of size x*y displaying a random set of images from the defined path')
    parser.add_argument('-p', '--path', type = str, required = True, help = 'path to image folder')
    parser.add_argument('-x', '--x', type = int, required = True, help = 'number of images in x ')
    parser.add_argument('-y', '--y', type = int, required = True, help = 'number of images in y ')
    parser.add_argument('-f1', '--figsize1', type = int, default = 10, help = 'display size [Default: %(default)s]')
    parser.add_argument('-f2', '--figsize2', type = int, default = 10, help = 'display size [Default: %(default)s]')
    args = parser.parse_args()

    show_random_imgs(args.path, args.x, args.y, figsize = (args.figsize1, args.figsize2))
