from typing import Any, Optional, List

import numpy as np
import pandas as pd
from astropy.wcs import WCS
from regions import PixCoord, EllipsePixelRegion
from astropy.coordinates import Angle


def image_preprocessing(image: np.ndarray, field: str) -> np.ndarray:
    """Example preprocessing function for basic images.

    Args:
        image (np.ndarray): image
        field (str): Not Implemented here, but will be passed.

    Returns:
        np.ndarray: Squeezed image. I.e. removed empty axis.
    """
    return np.squeeze(image)


def wcs_preprocessing(wcs, field: str):
    """Example preprocessing function for wcs (world coordinate system).

    Args:
        wcs: Input wcs.
        field (str): field name matching the respective wcs.

    Returns:
        Altered wcs.
    """
    if field in ["COSMOS"]:
        return (wcs.dropaxis(3).dropaxis(2),)
    elif field in ["XMMLSS"]:
        raise UserWarning(
            f"This may cause issues in the future. It is unclear where header would have been defined."
        )
    else:
        return wcs


def catalogue_preprocessing(
    df: pd.DataFrame, random_state: Optional[int] = None
) -> pd.DataFrame:
    """Example Function to make preselections on the catalog to specific
    sources meeting given criteria.

    Args:
        df (pd.DataFrame): Data frame containing catalogue information.
        random_state (Optional[int], optional): Random state seed. Defaults to None.

    Returns:
        pd.DataFrame: Subset catalogue.
    """
    # Only consider resolved sources
    df = df.loc[df["RESOLVED"] == 1]

    # Sort by S_INT (integrated flux)
    df = df.sort_values("S_INT", ascending=False)

    # Only consider unique islands of sources
    # df = df.groupby("ISL_ID").first()
    df = df.drop_duplicates(subset=["ISL_ID"], keep="first")

    # Sort by field
    # df = df.sort_values("field")

    return df.reset_index(drop=True)


def ellipse_to_box(major_axis: float, minor_axis: float, angle: float):
    """Calculate the size of the smallest rectangle that contains an ellipse defined by width, height and angle.

    Args:
        major_axis (float): The major axis of the ellipse.
        minor_axis (float): The minor axis of the ellipse.
        angle (float): The angle of the major axis in degrees.

    Returns:
        Tuple[int, int]: The size of the smallest rectangle in pixels that contains the ellipse. Height and width.
    """
    reg = EllipsePixelRegion(
        center=PixCoord(0, 0),
        width=major_axis,
        height=minor_axis,
        angle=Angle(angle, "deg"),
    )
    return reg.bounding_box.shape


def calculate_cutout_size(
    df,
    cutout_size_name: List[str] = ["cutout_height", "cutout_width"],
    major_axis: str = "Composite_Width",
    minor_axis: str = "Composite_Size",
    angle: str = "Composite_PA",
    scaling: float = 1.0,
    square: bool = False,
):
    """Calculate the cutout size of the source. The cutout size is defined as the size of the smallest rectangle that contains the source defined by the major and minor axis and the position angle of an encapsulating ellipse.

    Args:
        df (pd.DataFrame): DataFrame containing the source information.
        cutout_size_name (List[str], optional): Name of the columns to write the cutout size to. Defaults to ["cutout_height", "cutout_width"].
        major_axis (str, optional): Name of the column containing the major axis of the source. Defaults to "Composite_Width".
        minor_axis (str, optional): Name of the column containing the minor axis of the source. Defaults to "Composite_Size".
        angle (str, optional): Name of the column containing the angle of the respective major axis position angle of the source in degrees. Defaults to "Composite_PA".
        scaling (float, optional): Scaling factor applied to the cutout axes. Defaults to 1.0.
        square (bool, optional): If True, the cutout size will be a square with maximum size of the respective rectangle. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame containing the source information with the cutout size columns appended.
    """
    # Select rows where the major and minor axis are not nan
    df = df.dropna(subset=[major_axis, minor_axis, angle])
    # Drop zeros
    df = df[(df[major_axis] != 0) & (df[minor_axis] != 0)]
    sizes = df.apply(
        lambda x: ellipse_to_box(x[major_axis], x[minor_axis], x[angle]), axis=1
    )
    if square:
        sizes = sizes.apply(lambda x: (max(x), max(x)))
    df[[*cutout_size_name]] = pd.DataFrame(sizes.tolist(), index=df.index) * scaling
    return df[[*cutout_size_name]]
