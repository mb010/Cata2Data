from typing import Any, Optional

import numpy as np
import pandas as pd
from astropy.wcs import WCS


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
        wcs = WCS(header, naxis=2)  # This surely causes a bug right?
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
