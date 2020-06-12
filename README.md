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
  - 0: Flag by default, starts in block **patch_division**
  - 1: Starts in block **detect_dab**. Loads file `list_{level}_{ts}.p`.
  - 2: Starts in block **feature_extraction**. Loads files `list_positive_{level}_{ts}.p`
  and `class_{level}_{ts}.p`.
  - 3: Starts in block **cluster_division**. Loads files `features_{feature_method}_level{level}.p`
  and `class_{level}_{ts}.p`.
  - 4: Starts in visualization block. Loads file `class-{feature_method}-{level}-{method}.p`  
- **Device**: Choose the GPU device to use, "0" or "1" (string - "0" by default)
- **Jobs**: Number of CPU jobs to be used in parallelized parts of the code
- **-b**: In case there are too many patches, it is possible to run the
feature extraction in batches (feature_extraction_batch)

## Blocks

### Patch division

``` shell
python patch_division.py [-h] [-s SLIDES] [-o OUTPATH] [-l LEVEL]
                         [-ts TILE_SIZE] [-tr TISSUE_RATIO] [-j JOBS]
```

This block gets the slides with the format PDL1.mrxs found in `Slides` path and
divides them in patches. It uses the package OpenSlide and DeepZoom to open each
.mrxs slide and get to the image pyramid level to extract the patches. This
block will return and save:
- A slide list which contains all the slides which are divided plus their path.
This will be saved in `outpath` as `list_{level}_{ts}.p` and returned by the
function
- A preview of each slide, saved in `outpath` as `{slidename}.png`
- A folder containing the patches. The folder has the same name as the slide,
and the patches are saved with the format `{slide}#{n}-level{}-{i}-{j}.jpg` (
being n the index of the patch and i, j its coordenates in x, y).

The parameters required are:
- The path to the folder where all slides are saved.
- The outpath
- The deepzoom level desired (default 16)
- Tile size and tissue ratio (224 and 0.5 respectively)
- Number of jobs to be used in parallel

***Note about Pyramid levels and DeepZoom***

While normally level 0 of the pyramid holds the higher resolution, deepzoom levels
are ordered from smaller to bigger. Therefore, level 0 has dimensions 1x1 pixels
and higher levels grow in resolution until the complete image resolution.

![](https://github.com/pilarOrtega/pdl1/blob/master/images/patch_division.png)

In this example we display the preview of a slide in level 13 (which is level 5
of OpenSlide). The number of patches is 13x29, each one covering a surface of
224x224 pixels. The red patches are those which do not include any information
and are not saved (their tissue ratio is lower than the defined). The table
shows a correspondence between DeepZoom level, size and number of patches.

Functions:
- **get_patches**: Given a slide, extracts the patches of the given level and
size and saves in folder those which fulfil the conditions.
- **patch_division**: Main function, calls `get_patches` recursively for all
slides


### DAB detect

``` shell
python detect_dab.py [-h] [-l LIST_SLIDES] [-o OUTPATH] [-t THRESHOLD]
                     [-j JOBS]
```

This block detects the presence of DAB tinction in the patches. It is based on
the scikimage.color function `rgb2hed`. The function reads all the images in
the list obtained from the previous block, transform them to HED color dimension
(Hematoxilyn-Eosin-DAB) and gets the histogram for the DAB channel.

Using this histogram, the algorithm discriminates between the patches with DAB
presence and those without any DAB tinction. The sensitivity is adjusted by
setting a threshold level chosen empirically.

This block will return and save:
- A classifier list which is also stored in `outpath` as `class_{level}_{ts}.p`.
This list has a length equal to the number of slides. Each element on this list
is composed by the slidename, the path to the slidefolder (folder containing the
patches) and an array of shape nx4, being n the number of patches of the slide.
In the last column of the array, 0/1 marks the absence or presence of DAB
staining.
![](https://github.com/pilarOrtega/pdl1/blob/master/images/class{level}_{ts}.png)
- A list with all the paths of the patches which show DAB staining. This list is
also stored in `list_positive_{level}_{ts}.p`.

The parameters required are:
- The slide list
- The outpath
- The threshold for DAB detection (default 85)
- Number of jobs to be used in parallel

Functions:
- **dab**: Detects the presence of DAB staining in an image. Returns a boolean.
- **dab_division**: Divides a list of patches into positive or negative to DAB.
- **detect_dab_delayed**: Gets the dab division of patches and completes the
classification array.
- **detect_dab**: Main function, calls `detect_dab_delayed`in parallel for each
slide.

### Feature extraction

### Cluster division

###
