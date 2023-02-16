# Cata2Data
Use FITS catalogues, FITS image data and astropy to manipulate your data into a dynamically read data set!

See the [example.ipynb notebook](https://github.com/mb010/Cata2Data/blob/main/example.ipynb) to see how you can easily load in a number of data sets into a dynamically memory loaded iterable data set.

# Installation
With this package at `ROOT_PATH/Cata2Data` on your deveice, install using pip:

```bash
pip install -e ROOT_PATH/Cata2Data
```

# Quick Introduction
The *quickest* introcution:
```python
from cata2data import CataData
from torch.utils.data import DataLoader

field_names = ["A", "B"]
catalogue_paths = ["CAT_A.fits", "CAT_A.fits"]
image_paths     = ["IMG_A.fits", "IMG_B.fits"]

data = CataData(
    catalogue_paths=catalogue_paths,
    image_paths=image_paths,
    field_names=field_names
)

dataloader = DataLoader(data, batch_size=64, shuffle=True)
```

# Features
See the doc strings for detailed notes on all of the parameters which CataData accepts. Specifically, consider the cutout size, memmory mapping (`mmap`) and the various pre-processing options to match your needs.

> :warning: Note that currently catalogues are indexed through their `"RA"`" and `"DEC"` columns. Use the `catalogue_preprocessing` parameter to correctly name the columns until this has a better fix.

# Contributing
[Open an issue](https://github.com/mb010/Cata2Data/issues) and let us know what sort of issue you are experiencing.

[Open a pull request](https://github.com/mb010/Cata2Data/pulls) if you have added functionality or fixed a bug.


# Conceptual Workflow
CataData takes in fields of images and catalogues. Catalogues are merged into one dataframe and labelled with their respective field names. The length of CataData objects is the length of that dataframe. Entries are indexed through the dataframe and samples are cutout from the respective image using the units provided in the "RA" and "DEC" columns of the catalogue.

If catalogued features are needed to manipulate the iamges, we recommend using an image processing wrapper around CataData objects. I.e. a function like: `image_postprocessing(catadata_instance, index) -> np.ndarray` which calls the `catadata_object[index]` and manipulates the resulting image as required before returning it.