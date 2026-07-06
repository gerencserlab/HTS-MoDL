

# HTS-MoDL
HTS adaptation of MoDL mitochondrial segmentation and function prediction in live-cell images with deep learning.
This repository provides MoDL Server for Image Analyst MKII. MoDL server fast calling mitochondrial segmentation without relaunching Python environment for every image (series). This can be used with any Image Analyst MKII pipeline using the API call (http://localhost:5002/segment).

To cite authors of MoDL: Ding, Yang, Jintao Li, Jiaxin Zhang, et al. “Mitochondrial Segmentation and Function Prediction in Live-Cell Images with Deep Learning.” Nature Communications 16, no. 1 (2025): 743. https://doi.org/10.1038/s41467-025-55825-x.

## List of pipelines
* [Install MoDL Environment](install_MoDL_environment.md)
* [Start MoDL Server](MoDL_server.md)
* [Segment mitochondria with MoDL](Segment_mitochondria_with_MoDL.md)

## Installation
1. Clone this git in [Image Analyst MKII](https://www.imageanalyst.net/download.html) by Edit/Download and Manage Pipelines from GitHub. 
2. Press the "< > Code" button [above in this page](https://github.com/gerencserlab/HTS-MoDL/) and copy the URL of this git.
3. Paste the URL in the URL field in the Connect to Git window in Image Analyst MKII.
4. Press Download.
5. The pipelines deposited here will appear in the middle section of the Pipelines main menu.
6. Dowload retrained model [U-RNet+_1x_50epochs.hdf5](https://github.com/gerencserlab/HTS-MoDL/releases/download/U-RNet%2B_1x_50epochs/U-RNet+_1x_50epochs.hdf5)
7. Install Python 3.10 from python.org
8. [Install CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)
9. [Install CuDNN 8](https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-8.9.7.29_cuda11-archive)
10. Add CUDA and CuDNN to PATH
11. [Install MoDL Environment](install_MoDL_environment.md)
12. In every session, MODL Server need to be started with running this pipeline: [Start MoDL Server](MoDL_server.md)


## Description
MoDL is a deep learning-based software package for precise mitochondrial segmentation and function prediction in live-cell images, and allows for visualization and outputs detailed data on mitochondrion morphology features and functionality. In this fork of the original repository we adapted MoDL to our high-throughput screening workflows.

## Changes to segmentation prediction
 * Arbitrary image input size
 * Image upscaling to work on different resolution images
 * Smooth stitching
 * Command line arguments
 * Flask server / service operation for Image Analyst MKII-based workflows    

## Command line usage

```
.\.venv\Scripts\python.exe .\MoDL_seg\segment_predict_flexible.py `
  --input .\testraw\your_image.tif `
  --weights U-RNet+ `
  --overlap 64 `
  --scale 2 `
  --output-original-size
```

The `--weights` argument selects the segmentation model weights. The default is `U-RNet+.hdf5`. If only a filename is provided, MoDL looks for it in the project `.\model\` folder; the `.hdf5` extension is appended if omitted. A full path can also be supplied.

Examples:
```
--weights U-RNet+
--weights U-RNet+.hdf5
--weights C:\models\custom_model.hdf5
```

## API usage (POST http://127.0.0.1:5002/segment)
```
python.exe .\segment_predict_server.py

Invoke-RestMethod -Uri http://127.0.0.1:5002/segment -Method Post -Form @{
  "--dir" = "C:\0Git\MoDL\testraw\your_image.tif"
  "--overlap" = "64"
  "--scale" = "1.0"
  "--threshold" = "0.7"
  "--batch-size" = "1"
  "--weights" = "U-RNet+"
}
```

## API fields
```
--dir
--weights
--model
--overlap
--scale
--threshold
--batch-size
--output-original-size
```

`--model` is kept as a backward-compatible alias for `--weights`.



### Modified original description
***

## Table of Contents
 * Executable file
 * Requirements
 * Installation
 * Usage
 * Data Preparation
 * Model Training
 * Model Prediction
* Contributing
 * License

***

## Executable file
The executable files and usage instructions for MoDL can be found at "https://zenodo.org/records/10889134". The detailed procedure can be found in the ***operating process.doc*** file within the ***MoDL_OBP*** folder.

## Requirements

MoDL is built with Python and Tensorflow. Technically there are no limits to the operation system to run the code, but Windows system is recommended, on which the software has been tested. The inference process of the MoDL can run using the CPU only, but could be inefficiently. A powerful CUDA-enabled GPU device that can speed up the inference is highly recommended.

CUDA 11:
https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local
CuDNN :
https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/
cudnn-windows-x86_64-8.9.7.29_cuda11-archive


Both CUDA 11 and cuDNN 8 dll files must be in the path before startin Image Analyst MKII. Adjust if needed and execute setpath-programdata.ps1
Extract the cuDNN bin/*.dll files in the path in setpath-programdata.ps1.


The inference process has been tested with:

 * Windows 11 (version 23H2)
 * Python 3.10 works (64 bit)
 * tensorflow 2.5.0
 * 11th Gen Intel(R) Core(TM) i7-11700 @ 2.50GHz
 * NVIDIA GeForce RTX 3060 Ti, 3090, 4090, 5080 works

***

## Installation

1. Install python 3.8.0 
2. (Optional) If your computer has a CUDA-enabled GPU, install the CUDA and CUDNN of the proper version. Have them in the PATH.
3. The directory tree has been built. Download the MoDL_main.zip and unpack it, or clone the repository: 
```
git clone https://github.com/gerencserlab/HTS-MoDL
```

4. Open the terminal in the MoDL directory, install the required dependencies using pip:

```
pip install -r requirements.txt
```


The installation takes about 10 minutes in the tested platform. The time could be longer due to the network states.

*** 

##Usage
To use this project, follow these steps:

(1) For mitochondrial segmentation part

1. Prepare the training data by running ***MoDL_seg/data_load.py***. 
You will need to prepare original mitochondrial images and ground truth to the *' deform/train '* and *' deform/label '* directories of the original MoDL demo. 

2. You can directly use our pre-trained model ***U-RNet+*** to predict [https://zenodo.org/records/10889134], You will need to download it and unzip it to the *' model '* directory of the original MoDL demo.

3. (Optional) Train the model by running ***MoDL_seg/train.py***, then you will get a model for super-resolution microscopy images segmentation of mitochondrial, and the trained model will be saved in the *' model '* directory of the original MoDL demo.

4. Prepare the test images and use the trained model to make segmentation by running ***MoDL_seg/segment_predict.py***. 
You will need to prepare the test images to the *' testraw '* directory of the original MoDL demo. After prediction, the predicted segmentations and their pseudo-color implementation are stored separately in the *' final_results/bw '* and *' final_results/pseudo '* directories of the original MoDL demo.

(2) For mitochondrial function prediction part

1. Prepare the training data by running morphology_analysis.py.
You need to prepare the training set, segment it into 8-bit images using MoDL, and store them in the  *'final_results/bw '* directory. Then, use ***morphology_analysis.py*** to extract the morphological features of these images and generate a ***.csv*** file, saving in the *'final_results/bw/512x512_pixels'* directory. Taking *'deform/train/function_pre/U87 cell.csv'* as an example.

2. You can directly use our pre-trained model to predict [https://zenodo.org/records/10889134], You will need to download it and unzip it to the *' model '* directory of the original MoDL demo.

3. (Optional) Train the model by running ***MoDL_pre/train.py***, then you will get models for mitochondrial function prediction, and the trained model will be stored in the *' model '* directory of the original MoDL demo.

4. Prepare the test images and use the trained model to make predictions by running ***MoDL_pre/function prediction.py***. Here, we have prepared data for five cell lines: HeLa, HepG2, U87, L02, and 143B. After running ***function_prediction.py***, you will be prompted to input the name of the cell line. The predictions will be stored in the ***function_predictions.csv*** file within the *'final_results'* directory. 

***
##**A specific file description are as follows:**
##Data Preparation
1. To reduce computational cost and improves training efficiency, the original full 2048×2048 pixels images were cropped into multiple 512×512 pixels patches.

2. Place the training images (512x512 pixels) in the *' deform/train '* directory for mitochondrial segmentation. Place the training ***.csv*** file in the *' deform/function_pre '* directory for mitochondrial function prediction. 

3. Place the corresponding labels in the *' deform/label '* directory, run the ***data_load.py*** to convert the images and labels into .npy format.


##Model Training
1. Run the ***MoDL_seg/train.py*** to train the model for segmentation. Run the ***MoDL_pre/train.py*** to train the model for function prediction.

2. The trained model will be saved in the *'model'* directory and named ***U-RNet+.hdf5*** for segmentation. The ***.pkl*** file will be saved in the *'model'* directory for function prediction.

   Segmentation prediction can select a different `.hdf5` weights file with `--weights`. Filename-only values are resolved relative to the project *'model'* directory, and `.hdf5` is appended if it is omitted. Full paths are also accepted.

3. The training progress and performance metrics will also be saved in the *'model'* directory after training.


##Model Prediction
1. Place the test images (2048x2048 pixels) to be segmented in the *' testraw '* directory.

2. Run the ***MoDL_seg/segment_predict.py*** to make segmentation using the trained model.

3. The predicted segmentations of patches in three ways (4×4 patches, 4×3 patches, 3×4 patches). The 4×4 patches were used to stitch together a complete 2048x2048 pixels resolution image, while the 4×3 and 3×4 patches were used to fill the seams produced during the stitching process. The pseudo-color implementation are stored separately in the corresponding *' results/results_xx/bw '* and *' results/results_xx/pseudo '* directories.

4. The final merged segmentations of three ways and their pseudo-color implementation are stored separately in the *' final_results/bw '* and *' final_results/pseudo '* directories.

5. Run the ***MoDL_pre/function_prediction.py*** script to generate the ***function_prediction.csv*** results file and stored in the `'final_results'` directory.

##Contributing
Contributions to this project are welcome! 

Here are a few ways you can contribute:
Report bugs or suggest improvements by creating a new issue.
Implement new features or fix existing issues by creating a pull request.


##License
This project is covered under the GNU General Public 3.0 License.



