# PDL1 stage project

## Description

Code for automatic clustering of PD-L1 IHC pathology slides. This code will

## Usage

The command for the complete execution of the program (including the 4 main
blocks + the visualization blocks) is:

``` shell
python test.py [-s SLIDES] -o OUTPATH [-l LEVEL] [-tr TISSUE_RATIO]
               [-ts TILE_SIZE] [-f FEATURE_METHOD] [-n N_DIVISION]
               [--flag FLAG] [-d DEVICE] [-j JOBS] [-b]
```
The optional arguments to give are:

- **Slides**: It is the path to the folder in which the .mrxs slides will be stored.
***IMPORTANT*** Only slides following the expression `PDL1.mrxs` will be fed to
the pipeline.
- **Outpath**: Path to the folder where all the results will be saved. This
includes a folder for each slides including all individual patches, .p files
including intermediate results from different blocks, and .png files containing
the summary images for result preview.
- **Level**: Refers to the zoom level in which the patches will be extracted
(refer to deepzoom). By default, this level is set to 16. 
- **Tissue ratio**:
- **Tile size**:
- **Feature method**:
- **N division**:
- **Flag**:
- **Device**:
- **Jobs**:
- **-b**:
## Blocks

### Patch division

### DAB detect

### Feature extraction

### Cluster division

###
