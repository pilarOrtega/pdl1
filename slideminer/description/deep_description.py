"""
Slide description using deep neural networks.

For now it only uses pre-built architectures with keras.
"""
# coding: utf8
import os
# pyslideseg has no remote for now, so I re-implement the slide2csv procedure.
# from pyslideseg.patchdesc import slide2csv
from keras.applications.xception import preprocess_input as preproc_xce
import numpy
from openslide import OpenSlide
import csv
from pysliderois.tissue import slide_rois
from .networks import Encoder
from skimage.io import imread


def slidePatches2csv(input_slide_folder,
                     output_csv_path,
                     encoder,
                     patchsize,
                     batchsize,
                     weights):
    """
    Describe a slide from already extracted patches.

    Write description into a csv file.
    """
    # Check input slide path
    if not os.path.exists(input_slide_folder):
        raise FileNotFoundError(input_slide_folder)
    slidename = os.path.basename(input_slide_folder)
    model_name = encoder.network_name
    # Instantiate slide object
    print("Processing file: {}".format(input_slide_folder))
    csv_columns = ["Slidename"]
    csv_columns.append("XPosition")
    csv_columns.append("YPosition")
    for k in range(encoder.model.output.shape[1]._value):
        csv_columns.append('{}_feature_{}'.format(model_name, k))

    with open(output_csv_path, "w") as csvfile:
        writer = csv.DictWriter(csvfile, csv_columns)
        writer.writeheader()
        xlist = []
        ylist = []
        imlist = []
        for imname in os.listdir(input_slide_folder):
            if imname.endswith(".png"):
                patch = imread(os.path.join(input_slide_folder, imname))
                imnamebase, _ = os.path.splitext(imname)
                x, y = imnamebase.rsplit("_", 2)[1::]
                if len(xlist) == batchsize:
                    xlist = []
                    ylist = []
                    imlist = []
                xlist.append(x)
                ylist.append(y)
                imlist.append(patch)
                if len(xlist) == batchsize:
                    samples = encoder.predict(preproc_xce(numpy.array(imlist)))
                    for sample, feature_vector in enumerate(samples):
                        curr_row = {"Slidename": slidename,
                                    "XPosition": xlist[sample],
                                    "YPosition": ylist[sample]}
                        for i, val in enumerate(feature_vector):
                            curr_row["{}_feature_{}".format(model_name, i)] = val
                        writer.writerow(curr_row)
                    xlist = []
                    ylist = []
                    imlist = []
        # if an incomplete batch remains
        if len(xlist):
            samples = encoder.model.predict(preproc_xce(numpy.array(imlist)))
            for sample, feature_vector in enumerate(samples):
                curr_row = {"Slidename": slidename,
                            "XPosition": xlist[sample],
                            "YPosition": ylist[sample]}
                for i, val in enumerate(feature_vector):
                    curr_row["{}_feature_{}".format(model_name, i)] = val
                writer.writerow(curr_row)


def slide2csv(input_slide_path,
              output_csv_path,
              model_name,
              patchsize,
              level,
              batchsize,
              weights):
    """
    Describe a slide with its patches.

    Write description into a csv file.
    """
    # Check input slide path
    if not os.path.exists(input_slide_path):
        raise FileNotFoundError(input_slide_path)
    # Instantiate Encoder object
    encoder = Encoder(model_name, "output", patchsize)
    if weights != "":
        if not os.path.exists(weights):
            raise FileNotFoundError("Can't find weight file: {}".format(weights))
        else:
            encoder.load_weights(weights)
    # Instantiate slide object
    print("Processing file: {}".format(input_slide_path))

    slide = OpenSlide(input_slide_path)
    slidename = os.path.splitext(os.path.basename(input_slide_path))[0]

    csv_columns = ["Slidename"]
    csv_columns.append("XPosition")
    csv_columns.append("YPosition")
    for k in range(encoder.model.output.shape[1]._value):
        csv_columns.append('{}_feature_{}'.format(model_name, k))

    with open(output_csv_path, "w") as csvfile:
        writer = csv.DictWriter(csvfile, csv_columns)
        writer.writeheader()
        xlist = []
        ylist = []
        imlist = []
        for patch, coords in slide_rois(slide, level, patchsize, patchsize):
            x, y = coords
            if len(xlist) == batchsize:
                xlist = []
                ylist = []
                imlist = []
            xlist.append(x)
            ylist.append(y)
            imlist.append(patch)
            if len(xlist) == batchsize:
                samples = encoder.predict(preproc_xce(numpy.array(imlist)))
                for sample, feature_vector in enumerate(samples):
                    curr_row = {"Slidename": slidename,
                                "XPosition": xlist[sample],
                                "YPosition": ylist[sample]}
                    for i, val in enumerate(feature_vector):
                        curr_row["{}_feature_{}".format(model_name, i)] = val
                    writer.writerow(curr_row)
                xlist = []
                ylist = []
                imlist = []
        # if an incomplete batch remains
        if len(xlist):
            samples = encoder.model.predict(preproc_xce(numpy.array(imlist)))
            for sample, feature_vector in enumerate(samples):
                curr_row = {"Slidename": slidename,
                            "XPosition": xlist[sample],
                            "YPosition": ylist[sample]}
                for i, val in enumerate(feature_vector):
                    curr_row["{}_feature_{}".format(model_name, i)] = val
                writer.writerow(curr_row)


def describe_slide_folder(inputfolder,
                          outputfolder,
                          level=1,
                          patchsize=299,
                          model_name="xception",
                          device="0",
                          batchsize=1000,
                          weights="",
                          slidetype=".mrxs",
                          rewrite=True):
    """
    Cut slides in folder into patches and describe them.

    ****************************************************
    """
    # set the visible GPU devices
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    # first create the output directory if does not exist
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)

    # now get list of input files
    for name in os.listdir(inputfolder):
        # check whether file is a slide
        if name.endswith(slidetype):
            # process the slide
            slidebasename, _ = os.path.splitext(name)
            slideoutname = slidebasename + ".csv"
            outputfile = os.path.join(outputfolder, slideoutname)
            inputfile = os.path.join(inputfolder, name)
            if rewrite:
                slide2csv(inputfile,
                          outputfile,
                          model_name,
                          patchsize,
                          level,
                          batchsize,
                          weights)
            else:
                if not os.path.exists(outputfile):
                    slide2csv(inputfile,
                              outputfile,
                              model_name,
                              patchsize,
                              level,
                              batchsize,
                              weights)


def describe_slidePatch_folder(inputfolders,
                               outputfolder,
                               level=1,
                               patchsize=299,
                               model_name="xception",
                               device="0",
                               batchsize=1000,
                               weights="",
                               rewrite=True):
    """
    Describe patches in slide folders.

    **********************************
    """
    # set the visible GPU devices
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    # first create the output directory if does not exist
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)

    # Instantiate Encoder object
    encoder = Encoder(model_name, "output", patchsize)
    if weights != "":
        if not os.path.exists(weights):
            raise FileNotFoundError("Can't find weight file: {}".format(weights))
        else:
            encoder.load_weights(weights)

    # now get list of input files
    for inputfolder in inputfolders:
        suboutputfolder = os.path.join(outputfolder, os.path.basename(inputfolder))
        # first create the output directory if does not exist
        if not os.path.exists(suboutputfolder):
            os.makedirs(suboutputfolder)
        for slidebasename in os.listdir(inputfolder):
            # here slidebasename is also a folder,
            # it must contain png files
            slideoutname = slidebasename + ".csv"
            outputfile = os.path.join(suboutputfolder, slideoutname)
            inputslide = os.path.join(inputfolder, slidebasename)
            if rewrite:
                slidePatches2csv(inputslide,
                                 outputfile,
                                 encoder,
                                 patchsize,
                                 batchsize,
                                 weights)
            else:
                if not os.path.exists(outputfile):
                    slidePatches2csv(inputslide,
                                     outputfile,
                                     encoder,
                                     patchsize,
                                     batchsize,
                                     weights)
