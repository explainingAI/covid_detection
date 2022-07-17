# Covid Detection

## uiblungs package

This package contains the logic to split and slice lung images with masks.

It can be executed as a module or as an imported python package using the Splitter class.

Module execution syntax:

`` python3 covid_detection split image_path mask_path output_path``

`` python3 covid_detection slice image_path mask_path output_path --n_slices``
--n_slices default value is 4

## Vfeautres package

This package calculates the vfeatures for the lung image in different areas

It can be executed as a module or as an imported python package using the Calculator class.

Module execution syntax:

`` python3 covid_detection calculate_features image_path mask_path output_path --n_colors --distances --angles --n_slices``


