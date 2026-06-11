# Segment mitochondria with MoDL
## Description
This pipeline uses MoDL to segment mitochondria a single channel. 


## Parameters
| # | Name | Type | Description |
|---|------|------|-------------|
| 0 | Image type for intensity scaling | Text | Only fluorescence images are rescaled, brightfield ones are used as are. Inverted brightfield images will be rescaled. |
| 1 | SRRF: Scale maximum (percentile) | Real | This percentile of the image histogram sets the intensity value where the maximum of the Look Up Table (LUT) is scaled. Use -1 to override this with fixed value set below at "Max value". |
| 2 | SRRF: Gamma | Real | A gamma value >1 makes image supralinearly brighter, a gamma value <1 makes the image sublinearly darker. |
| 3 | SRRF: Smooth factor | Real | Wiener filter noise level (0-1). Higher value provides more smoothing. Set zero for no smoothing. |
| 4 | Path and filename or URL (*.exe,*.bat,"command && command") | Text | filename, without arguments. Filename with full path, or filename only if it is in the environment path. Alternatively, use one or more command prompt expressions with arguments concatenated by && and the whole expression quoted with ". |
| 5 | Argument 2 (--overlap) | Text | Argument entries will be interspersed with the "Argument" entry |
| 6 | Argument 3 (--scale) | Text | Argument entries will be interspersed with the "Argument" entry |
| 7 | Argument 4 (--batch-size) | Text | Argument entries will be interspersed with the "Argument" entry |
| 8 | Argument 5 (--output-original-size) | Text | Argument entries will be interspersed with the "Argument" entry |


## Structure
![structure](/img/Segment_mitochondria_with_MoDL.jpg)

[Image Analyst MKII](https://www.imageanalyst.net) pipeline - saved by V4.3.5 (build 1038)

