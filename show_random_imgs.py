from random import randint
import os
from matplotlib import pyplot as plt
import glob
import argparse
from skimage.io import sift, imread, imsave, imshow


def show_random_imgs(images, x, y, figsize=(10, 10), save_fig=False, name=''):
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
            k = randint(0, len(images) - 1)
            image_data = os.path.basename(images[k])
            image_slide = image_data.split('#')[0]
            image_slide = image_slide.split('_')[2]
            image_slide = image_slide.split('.')[3]
            image_number = image_data.split('#')[1]
            image_number = image_number.split('-')[0]
            im = imread(images[k])
            ax[i].imshow(im)
            ax[i].set_title(image_slide + '-' + image_number,
                            fontdict={'fontsize': 6, 'fontweight': 'medium'})
    title = os.path.basename(name) + ': ' + str(len(images))
    fig.suptitle(title, va='baseline')
    fig.tight_layout()
    plt.show()
    if save_fig:
        fig.savefig(name, bbox_inches='tight', dpi=fig.dpi)
    # plt.close(fig)
    return len(images)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Shows a grid of size x*y displaying a random set of images from the defined path')
    parser.add_argument('-p', '--path', type=str, help='path to image folder')
    parser.add_argument('-x', '--x', type=int, required=True,
                        help='number of images in x ')
    parser.add_argument('-y', '--y', type=int, required=True,
                        help='number of images in y ')
    parser.add_argument('-f1', '--figsize1', type=int,
                        default=10, help='display size [Default: %(default)s]')
    parser.add_argument('-f2', '--figsize2', type=int,
                        default=10, help='display size [Default: %(default)s]')
    parser.add_argument('-s', '--savefig',
                        action='store_true', help='saves image as png')
    args = parser.parse_args()

    image_paths = os.path.join(args.path, '*.jpg')
    image_list = glob.glob(image_paths)
    name = args.path
    name = os.path.split(name)
    cluster = 'cluster_{}.png'.format(name[1])
    name = os.path.join(name[0], cluster)
    if args.savefig:
        print('Save image in path ' + name)
        show_random_imgs(image_list, args.x, args.y, figsize=(
            args.figsize1, args.figsize2), save_fig=True, name=name)
    else:
        show_random_imgs(image_list, args.x, args.y,
                         figsize=(args.figsize1, args.figsize2))
