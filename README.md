# Cata2Data
Use FITS catalogues, FITS image data and astropy to manipulate your data into a dynamically read data set!

See the [example notebook](https://github.com/mb010/Cata2Data/blob/main/example.ipynb) to see how you can easily load in a number of data sets into a dynamically memory loaded iterable data set.

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
