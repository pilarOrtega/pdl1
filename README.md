# PDL1 stage project

## Description

Code for automatic clustering of PD-L1 IHC pathology slides.

## Usage

The command for the complete execution of the program (including the 4 main
blocks + the visualization blocks) is:

``` shell
python test.py [-s SLIDES] -o OUTPATH [-l LEVEL] [-tr TISSUE_RATIO]
               [-ts TILE_SIZE] [-f FEATURE_METHOD] [-n N_DIVISION]
               [-m METHOD] [--flag FLAG] [-d DEVICE] [-j JOBS] [-b]
```
The optional arguments to give are:

- **Slides**: It is the path to the folder in which the .mrxs slides will be stored.
***IMPORTANT*** Only slides following the expression `PDL1.mrxs` will be fed to
the pipeline.
- **Outpath**: Path to the folder where all the results will be saved. This
includes a folder for each slides including all individual patches, .p files
including intermediate results from different blocks, and .png files containing
the summary images for result preview. If the code is run with a flag different
than 0, it will search for intermediate results in this folder. ***Note***
Although patches are saved in this folder, it is not necessary to change the
location if they are going to be reused, since the path to each patches is saved
in intermediate files.
- **Level**: Refers to the zoom level in which the patches will be extracted
(refer to deepzoom). By default, this level is set to 16.
- **Tissue ratio**: By default set to 0.5, it is the percentage of tissue that
a patch needs to have to be saved and considered in the algorithm.
- **Tile size**: It is the number of pixels in the side of a patch, this is, if
set to 224 (default) patches will be of size 224x224. ***Room for optimization***
- **Feature method**: For now, there are 4 different feature extraction methods
that can be used, either in all RGB channels or only in the DAB channel - 8
possible feature methods in total,, namely *Dense, DenseDAB, Daisy,* and
*DaisyDAB,* which are classical methods and *Xception, XceptionDAB, VGG16* and
*VGG16DAB*, based on Xception and VGG16 pretrained nn respectively.
- **N division**: In the case of method TopDown, it is the number of cluster
subdivisions that will be made (by default set to 4). In case of BottomUp, this
parameter is not necessary and is set to 1.
- **Method**: BottomUp or TopDown. The first executes a K-Means clustering
with k=16 (with the aim of later regrouping clusters in a hierarchical way) while
TopDown executes k-means with k=2 recursively (up to 2**ndivision clusters)
- **Flag**: Integer from 0 to 4, it sets the start of the execution in a
specific block:
  0. Flag by default, starts in block **patch_division**
  1. Starts in block **detect_dab**. Loads file `list_{level}_{ts}.p`.
  2. Starts in block **feature_extraction**. Loads files `list_positive_{level}_{ts}.p`
  and `class_{level}_{ts}.p`.
  3. Starts in block **cluster_division**. Loads files `features_{feature_method}_level{level}.p`
  and `class_{level}_{ts}.p`.
  4. Starts in visualization block. Loads file `class-{feature_method}-{level}-{method}.p`  
- **Device**: Choose the GPU device to use, "0" or "1" (string - "0" by default)
- **Jobs**: Number of CPU jobs to be used in parallelized parts of the code
- **-b**: In case there are too many patches, it is possible to run the
feature extraction in batches (feature_extraction_batch)

## Blocks

### Patch division

### DAB detect

### Feature extraction

### Cluster division

###
