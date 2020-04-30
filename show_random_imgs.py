from random import randint
import os
from matplotlib import pyplot as plt
import glob
import argparse
from skimage.io import sift, imread, imsave, imshow


def show_random_imgs(images, x, y, figsize=(10, 10)):
    n = x * y
    fig, axes = plt.subplots(x, y, figsize=figsize, sharex=True, sharey=True)
    print('Number of images ' + str(len(images)))
    ax = axes.ravel()
    if len(images) <= n:
        for i in range(len(images)):
            im = imread(images[i])
            ax[i].imshow(im)
    else:
        for i in range(n):
            k = randint(0, len(images)-1)
            im = imread(images[k])
            ax[i].imshow(im)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Shows a grid of size x*y displaying a random set of images from the defined path')
    parser.add_argument('-p', '--path', type=str, help='path to image folder')
    parser.add_argument('-x', '--x', type=int, required=True, help='number of images in x ')
    parser.add_argument('-y', '--y', type=int, required=True, help='number of images in y ')
    parser.add_argument('-f1', '--figsize1', type=int, default=10, help='display size [Default: %(default)s]')
    parser.add_argument('-f2', '--figsize2', type=int, default=10, help='display size [Default: %(default)s]')
    args = parser.parse_args()

    image_paths = os.path.join(args.path, '*.jpg')
    image_list = glob.glob(image_paths)
    show_random_imgs(image_list, args.x, args.y, figsize=(args.figsize1, args.figsize2))
