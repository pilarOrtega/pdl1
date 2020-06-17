import numpy
import os

def cl_to_class(clusterlist, class_init):
    classifiers = []
    for s in class_init:
        n_samples = s[2].shape[0]
        c = numpy.zeros((n_samples, 5))
        c[:, : - 1] = s[2]
        classifiers.append((s[0], s[1], c))

    slide_list = [x[0] for x in classifiers]

    for im in clusterlist:
        image_name = os.path.basename(im[0])
        image_name = image_name.split('#')[1]
        number = image_name.split('-')
        number = int(number[0])
        slide_path = os.path.dirname(im[0])
        index_slide = slide_list.index(os.path.basename(slide_path))
        classifiers[index_slide][2][number][4] = im[1]

    return classifiers
