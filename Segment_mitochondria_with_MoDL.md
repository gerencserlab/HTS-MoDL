# Segment mitochondria with MoDL
## Description
This pipeline uses MoDL to segment mitochondria in a single channel. 
Download orignal model (super resolution) from: https://zenodo.org/records/6481291
Download retrained model (spinning disc confocal, 0.18um/pixel) from: https://github.com/gerencserlab/HTS-MoDL/releases/download/U-RNet%2B_1x_50epochs/U-RNet+_1x_50epochs.hdf5

To cite authors of MoDL: Ding, Yang, Jintao Li, Jiaxin Zhang, et al. “Mitochondrial Segmentation and Function Prediction in Live-Cell Images with Deep Learning.” Nature Communications 16, no. 1 (2025): 743. https://doi.org/10.1038/s41467-025-55825-x.


## Parameters
| # | Name | Type | Description |
|---|------|------|-------------|
| 0 | Path and filename or URL (*.exe,*.bat,"command && command") | Text | filename, without arguments. Filename with full path, or filename only if it is in the environment path. Alternatively, use one or more command prompt expressions with arguments concatenated by && and the whole expression quoted with ". |
| 1 | Argument 2 (--overlap) | Text | Overlap of 512x512 patches to avoid edge artifacts |
| 2 | Argument 3 (--scale) | Text | Up or down scaling to match training and inference resolution |
| 3 | Argument 4 (--batch-size) | Text | Batch size (mind your VRAM!) |
| 4 | Argument 5 (--output-original-size) | Text | Return original or scaled sized image |
| 5 | Argument 7 (--weights) | Text | Path to downloaded network weight *.hdf5 file |
| 6 | Minimum probability (0-255 range) | Real | Mean of pixel intensities in Image B. Use 0 for not checking |
| 7 | Minimum intensity (0-255 range) | Real | Mean of pixel intensities in Image B. Use 0 for not checking |


## Structure
![structure](/img/Segment_mitochondria_with_MoDL.jpg)

[Image Analyst MKII](https://www.imageanalyst.net) pipeline - saved by V4.3.6 (build 1042)

