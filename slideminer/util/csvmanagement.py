"""
Util functions for csv data management.

Merges csv files into one single files.
"""
# coding: utf8
import os
import csv
import numpy


def get_segmentation_info(segcsv):
    """
    Get information available in a csv segmentation file.

    Return number of description features and network architecture used
    for feature extraction.
    """
    with open(segcsv, "r") as f:
        reader = csv.DictReader(f)
        n_features = 0
        archi = ""
        for colname in reader.fieldnames:
            if "feature_" in colname:
                if n_features == 0:
                    archi = colname.split("_")[0]
                n_features += 1
        return n_features, archi


def segments_from_csv(csvseg):
    """
    Get segment dictionary from a csv segmentation file.

    Return the segments of the slide (label and descriptor).
    """
    n_features, archi = get_segmentation_info(csvseg)
    segments = dict()
    populations = dict()
    with open(csvseg, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(row["Label"])
            features = numpy.array([float(row["{}_feature_{}".format(archi, k)]) for k in range(n_features)])
            if label not in populations:
                populations[label] = 1
                segments[label] = features
            else:
                populations[label] += 1
                segments[label] += features
    for l in segments:
        segments[l] /= float(populations[l])

    return segments


def stack_segments_from_multiple_slides(inputfolder, outputfile):
    """
    Stack segments from multiple csv segmentation files into a single csv file.

    Every segment is stored with its in-slide label and a general label.
    Average description of the segment is also stored in the csv line.
    """
    segment_counter = 0

    # get the first csv in the folder
    csv_list = []
    for filename in os.listdir(inputfolder):
        name, ext = os.path.splitext(filename)
        if ext == ".csv" and not name.startswith("."):
            csv_list.append(os.path.join(inputfolder, filename))

    n_features, archi = get_segmentation_info(csv_list[0])
    columns = ["Slidename", "InSlideId", "GlobalId"]
    for label in range(n_features):
        columns.append("{}_feature_{}".format(archi, label))

    # create output csv file
    with open(outputfile, "w") as f:
        # create the writer
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        csv_counter = 1
        for csv_file in csv_list:
            print("Processing file {} / {}".format(csv_counter, len(csv_list)))
            # we will have to store slide names and InSlideLabel,
            # otherwise, we will not be able to get back to the images...
            segments = segments_from_csv(csv_file)
            print("File {} has {} segments.".format(os.path.basename(csv_file),
                                                    len(segments)))
            for inslidelabel, segfeatures in segments.items():
                slidename, _ = os.path.splitext((os.path.basename(csv_file)))
                line_dico = {"GlobalId": segment_counter,
                             "InSlideId": inslidelabel,
                             "Slidename": slidename}
                for k in range(n_features):
                    line_dico["{}_feature_{}".format(archi, k)] = segfeatures[k]
                writer.writerow(line_dico)
                segment_counter += 1
            csv_counter += 1


def global_by_local_by_name(stackcsv):
    """
    Produce a dictionary from a stack of csv segments.

    For every line, read the name, global and local id of the segment.
    """
    global_by_local_by_name = dict()
    # read the csv
    with open(stackcsv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            slidename = row["Slidename"]
            loc = row["InSlideId"]
            gl = row["GlobalId"]
            if slidename not in global_by_local_by_name:
                global_by_local_by_name[slidename] = {int(loc): int(gl)}
            else:
                global_by_local_by_name[slidename][int(loc)] = int(gl)
    return global_by_local_by_name


def read_segcsv(segcsv):
    """
    Iterate on a segmentation csv.

    *******************************
    """
    with open(segcsv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            loc = int(row["Label"])
            x = int(row["XPosition"])
            y = int(row["YPosition"])
            yield loc, x, y
