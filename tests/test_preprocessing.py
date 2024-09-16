import pytest
import numpy as np
import pandas as pd
from astropy.wcs import WCS


from cata2data import (
    image_preprocessing,
    wcs_preprocessing,
    catalogue_preprocessing,
    ellipse_to_box,
    calculate_cutout_size,
)


def test_image_preprocessing():
    image = np.random.rand(1, 5, 5)
    result = image_preprocessing(image, "test_field")
    assert result.shape == (
        5,
        5,
    ), "Image preprocessing failed to remove the correct axis."


def test_wcs_preprocessing_cosmos():
    wcs = WCS(naxis=4)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN", "FREQ", "STOKES"]
    result = wcs_preprocessing(wcs, "COSMOS")
    assert (
        result[0].naxis == 2
    ), "WCS preprocessing for COSMOS did not remove correct axes."


def test_wcs_preprocessing_xmmlss_warning():
    wcs = WCS(naxis=4)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN", "FREQ", "STOKES"]
    with pytest.raises(UserWarning):
        wcs_preprocessing(wcs, "XMMLSS")


def test_catalogue_preprocessing():
    data = {
        "RESOLVED": [1, 0, 1],
        "S_INT": [3.0, 1.0, 2.0],
        "ISL_ID": [1, 2, 1],
    }
    df = pd.DataFrame(data)
    result = catalogue_preprocessing(df)
    assert (
        0 not in result["RESOLVED"].values
    ), "Catalogue preprocessing did not filter correctly"
    assert (
        result.iloc[0]["S_INT"] == 3.0
    ), "Catalogue preprocessing did not sort correctly."
    assert (
        np.unique(result["ISL_ID"].to_numpy()).size == 1
    ), "Catalogue preprocessing did not handle duplicates correctly."


def test_ellipse_to_box():
    width = 10
    height = 5
    angle = 45
    result = ellipse_to_box(width, height, angle)
    assert (
        isinstance(result, tuple) and len(result) == 2
    ), "Ellipse to box did not return a tuple of correct size."
    assert (
        result[0] > 0 and result[1] > 0
    ), "Ellipse to box returned invalid dimensions."


def test_calculate_cutout_size():
    data = {
        "Composite_Width": [10, 20, np.nan],
        "Composite_Size": [5, 10, 15],
        "Composite_PA": [45, 60, 30],
    }
    df = pd.DataFrame(data)
    result = calculate_cutout_size(df)
    assert (
        result.shape[0] == 2
    ), "Calculate cutout size did not drop NaN values correctly."
    assert (
        result.iloc[0]["cutout_height"] > 0 and result.iloc[0]["cutout_width"] > 0
    ), "Calculate cutout size failed to compute correct dimensions."


def test_calculate_cutout_size_square():
    data = {
        "Composite_Width": [10, 20],
        "Composite_Size": [5, 10],
        "Composite_PA": [45, 60],
    }
    df = pd.DataFrame(data)
    result = calculate_cutout_size(df, square=True)
    assert (
        result["cutout_height"] == result["cutout_width"]
    ).all(), "Calculate cutout size square option failed."
