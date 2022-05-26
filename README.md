# Covid Detection

## Lung_image_splitter package

This package has two public modules:
- dataset_generator: generates a dataset with the coordinates of the zones to cut from the mask image
  - Input: Masks folder, output folder, (optional) number of images per lung
  - Output: csv with the coordinates for each image of the folder
- image_splitter: splits the image in n sub images using a mask and a lungs image.
  - Input: Masks folder, images folder, output folder, (optional) number of images per lung
  - Output: Sub images


The modules can be executed using command prompt:

`` python3 module_name arg1 arg2 arg3 argn``
