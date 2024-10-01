# LOTSS DR2 Data Class

This folder contains the utilities to produce a full dataloder for LOTSS DR2 using [Cata2Data](https://github.com/mb010/Cata2Data).
The dataloader was initially developed for a different project. It serves to highlight how powerful of a tool [Cata2Data](https://github.com/mb010/Cata2Data) can be.

# Quick walkthrough:
To start, create a local clone of this repository and navigate to this directory.

Install cata2data into your local environment (We recommend that you should use a [venv](https://docs.python.org/3/library/venv.html)).

To install Cata2Data, activate your virtual environment and run `pip install -e <path/to/Cata2Data/root_dir>`.

## Download Data

```bash
python data_scrapper.py --dir DIRECTORY_TO_SAVE_TO
```

if you want to just download one pointing (instead of all 841 pointings; 434 GB), then call it as a test:

```bash
python data_scrapper.py --dir DIRECTORY_TO_SAVE_TO --test
```

The `data_scrapper` script will download the image file. Next, you need to download the catalog directly from the website at [this link](https://lofar-surveys.org/public/DR2/catalogues/combined-release-v1.1-LM_opt_mass.fits) (3.9 GB). This dataloader is currently built to work with the [Radio-optical cross match catalog](https://lofar-surveys.org/dr2_release.html#:~:text=Radio%2Doptical%20crossmatch%20catalogue) described in [Hardcastle et al. 2023](https://arxiv.org/abs/2309.00102).

## Split the Catalog

```bash
python catalog_splitter.py --catalog_path PATH_TO_THE_FULL_CATALOG --image_paths PATH_TO_DIRECTORY_OF_IMAGES
```

This will take the full catalog and split it into one catalog per image and save those into the folder where each of those images is stored. This is what Cata2Data currently expects - lists of images and catalogs with equal length to use to construct a dataloader.

## Construct the dataset
A number of decisions have been made in the selection of sources etc, but in general everything is in [the data.py file](data.py). To run the code below you can run the `Create_LoTTSDataset.ipynb` [notebook in Colab](https://colab.research.google.com/github/mb010/Cata2Data/blob/main/examples/lotssdr2/Create_LoTTSDataset.ipynb).

```python
from data import LoTTSDataset
from torchvision.transforms import v2
import torch

transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32),
    v2.Resize(size=(64, 64)),
])

data = LoTTSDataset(
    data_folder="./data/lotssdr2/public", # Change this to where you saved your data
    cutout_scaling=1.5,
    transform=transforms,
)

for i in range(len(data)):
    if i > 10:
        break
    data.plot(i, contours=True, sigma_name="Isl_rms", min_sigma=2, title=data.df.iloc[i]["Source_Name"] + data.df.iloc[i]["S_Code"])

data.df.head()
```
