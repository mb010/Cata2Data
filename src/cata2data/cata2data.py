import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import astropy.units as units
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regions
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.units import Quantity
from astropy.wcs import WCS
from spectral_cube import SpectralCube, StokesSpectralCube
from pathlib import Path


class CataData:
    """A class taking fits catalogues and images and producing
    a data set for deep learning."""

    def __init__(
        self,
        catalogue_paths: Union[List[str], str],
        image_paths: Union[List[str], str],
        field_names: Union[List[Union[int, int]], str],
        cutout_width: Union[int, Quantity] = 32,
        memmap: bool = True,
        transform: Optional[Callable] = None,
        catalogue_preprocessing: Optional[Callable] = None,
        wcs_preprocessing: Optional[Callable] = None,
        fits_index_catalogue: int = 1,
        fits_index_images: int = 0,
        image_drop_axes: List[int] = [3, 2],
        origin: int = 1,
        spectral_axis: bool = False,
        stokes_axis: bool = False,
        return_wcs: bool = False,
    ) -> None:
        """Produces a deep learning ready data set from fits catalogues
        and fits images.

        Args:
            catalogue_paths (List[str] | str):
                Fits catalogue path(s) in matching order.
            image_paths (List[str] | str):
                Fits image path(s) in matching order.
            field_names (List[str  |  int] | str):
                Names of the field(s) in matching order.
            cutout_width (int, optional):
                Cut out pixel width. Defaults to 32.
            memmap (bool, optional):
                Whether to use memory mapping (dynamic reading of images into memory). Defaults to False.
            transform (Optional[Callable], optional):
                Transformations to use. Currently not implemented. Defaults to None.
            catalogue_preprocessing (Optional[Callable], optional):
                Function to apply to catalogues before use. Ideal for filtering to subsamples. Defaults to None.
            wcs_preprocessing (Optional[Callable], optional):
                Function applied to the astropy wcs object before selecting data from the images. Defaults to None.
            fits_index_catalogue (int, optional): Index used in self.open_fits call. Selects correct wcs entry for
            respective catalogues. Ordered appropriately with paths. Defaults to 1.
            fits_index_images (int, optional): Index used in self.open_fits call. Selects correct wcs entry for
            respective images. Ordered appropriately with paths. Defaults to 0.
            image_drop_axes (List[int], optional): Not Implemented. Defaults to [3,2].
            origin (int, optional): Wcs origin. Used in cutout to calculated wcs.world2pix coords. Defaults to 1.
            spectral_axis (bool): TO BE COMPLETED. Detaults to False.
            stokes_axis (bool): TO BE COMPLETED. Defaults to False.

        """
        self.catalogue_paths = (
            catalogue_paths if type(catalogue_paths) is list else [catalogue_paths]
        )
        self.image_paths = image_paths if type(image_paths) is list else [image_paths]
        self.field_names = field_names if type(field_names) is list else [field_names]
        self.fits_index_catalogue = fits_index_catalogue
        self.fits_index_images = fits_index_images

        # Checks
        self._verify_input_lengths()
        self._check_exists()

        self.catalogue_preprocessing = catalogue_preprocessing
        self.wcs_preprocessing = wcs_preprocessing

        # if transform is albumentations transform do ...
        self.transform = transform

        self.memmap = memmap
        self.origin = origin
        self.cutout_width = cutout_width
        self.spectral_axis = spectral_axis
        self.stokes_axis = stokes_axis
        self.return_wcs = return_wcs

        self.df = self._build_df()
        self.images, self.wcs = self._build_images(image_drop_axes)

    def __getitem__(self, index: int, force_return_wcs: bool = False) -> np.ndarray:
        """Gets the respective indexed item within the data set. Indexes from the built catalogue data frame.

        Args:
            index (int): Index of item to retrieve.
            return_wcs (bool, optional): Whether to return coordinates of object (returns as list of coordinates of length one). Defaults to False.

        Raises:
            ValueError: If index is larger than the size of the data set.

        Returns:
            np.ndarray: Cutout image with indexed coordinates at centre of the cutout (batch dimension first).
        """
        if index >= self.__len__():
            raise ValueError(f"Index out of range.")
        coords = self.df.iloc[index : index + 1][["ra", "dec"]].values
        field = self.df.iloc[index].field
        return_wcs = True if (self.return_wcs or force_return_wcs) else False
        return self.cutout(coords, field=field, return_wcs=return_wcs)

    def __len__(self) -> int:
        """Returns the length of the processed catalogue. Necessary for pytorch dataloaders.

        Returns:
            int: Data set length.
        """
        return len(self.df)

    def cutout(
        self, coords: np.ndarray, field: Union[str, int], return_wcs: bool = False
    ) -> Union[tuple, np.ndarray]:
        """Produces a set of images based on the provided
        coordinates and field.

        Args:
            coords (np.ndarray): Source coordinates.
            field (Union[str, int]): Field name or number.
            return_wcs (bool, optional): Whether to return the coordinate objects of the cutouts. Defaults to False.

        Returns:
            Union[tuple, np.ndarray]: Cutouts and coordinates (WCS) or cutouts.
        """
        ### Currently using:
        # https://docs.astropy.org/en/stable/nddata/utils.html#cutout-images
        # https://docs.astropy.org/en/stable/api/astropy.nddata.Cutout2D.html

        wcs = self.wcs[field]
        if "RADESYS" in wcs.to_header().keys():
            skycoord_coordinates = SkyCoord(
                ra=coords[:, 0] * units.deg,
                dec=coords[:, 1] * units.deg,
                frame=wcs.to_header()["RADESYS"].lower(),
            )
        else:
            skycoord_coordinates = SkyCoord(
                ra=coords[:, 0] * units.deg, dec=coords[:, 1] * units.deg
            )
            raise UserWarning(
                f"Images of field '{field}' may not be image files since the header does not contain a 'RADESYS' entry."
            )

        cutouts = []
        wcs_ = []
        for coord in skycoord_coordinates:
            if self.spectral_axis:
                region = regions.RectanglePixelRegion(
                    regions.PixCoord.from_sky(coord, wcs, 0, "wcs"),
                    self.cutout_width,
                    self.cutout_width,
                )
                if self.stokes_axis:
                    cutout = []
                    for component in self.images[field].components:
                        cutout_ = (
                            self.images[field]
                            ._stokes_data[component]
                            .subcube_from_regions([region])
                        )
                        cutout.append(cutout_.unmasked_data[:].value)
                        wcs_.append({component: cutout_.wcs})
                    cutouts.append(np.stack(cutout))
                else:
                    cutout_ = self.images[field].subcube_from_regions([region])
                    cutouts.append(cutout_.unmasked_data[:])
                    if return_wcs:
                        wcs_.append(cutout_.wcs)
            else:
                cutout = Cutout2D(
                    data=np.squeeze(self.images[field]),
                    position=coord,
                    size=(
                        self.cutout_width,
                        self.cutout_width,
                    ),  # Could be anything? Or just unconstrained?
                    wcs=self.wcs[field],
                    mode="partial",
                    fill_value=0,
                )
                wcs = cutout.wcs
                cutouts.append(cutout.data)
                if return_wcs:
                    wcs_.append(cutout.wcs)

        if return_wcs:
            return np.stack(cutouts), wcs_
        return np.stack(cutouts)

    def save_cutout(path: str, index: int, format: str = "fits") -> None:
        """Saves the cutout of the respective index.
        Args:
            path (str): Path which the file should be saved to.
            index (int): Catalogue index used to produce cutout.
            format (str): File out type. Defaults to 'fits' (coordinate system corrected header).
        """
        if self.spectral_axis:
            raise NotImplementedError
        cutout, wcs = self.__getitem__(index, force_return_wcs=True)
        wcs = wcs[0]  # unpack extract dimension
        cutout = cutout[0]
        if format == "fits":
            fits.writeto(path, data=cutout, header=wcs.to_header())
        else:
            raise NotImplementedError("Currently only fits format supported.")
        return

    def plot(
        self, index: int, contours: bool = False, sigma_name: str = "ISL_RMS"
    ) -> None:
        """Plot the source with the given index.

        Args:
            index (int): Index of the source to plot.
        """
        if self.spectral_axis:
            raise Warning(
                "Plotting not implemented for spectral cube data. Passing the plotting call."
            )
            return

        crosshair_alpha = 0.3
        image, wcs = self.__getitem__(index, force_return_wcs=True)
        image = np.squeeze(image[0])
        wcs = wcs[0]
        plt.subplot(projection=wcs)
        plt.imshow(image, origin="lower", cmap="Greys")
        plt.colorbar()
        if contours:
            plt.contour(
                image,
                levels=[self.df.iloc[index][sigma_name] * (3 + n) for n in range(3)],
                origin="lower",
            )
        plt.plot(
            (self.cutout_width // 2, self.cutout_width // 2),
            (0, self.cutout_width - 1),
            color="red",
            linewidth=2,
            ls="--",
            alpha=crosshair_alpha,
        )
        plt.plot(
            (0, self.cutout_width - 1),
            (self.cutout_width // 2, self.cutout_width // 2),
            color="red",
            linewidth=2,
            ls="--",
            alpha=crosshair_alpha,
        )
        plt.xlim(0, self.cutout_width - 1)
        plt.ylim(0, self.cutout_width - 1)
        plt.show()

    def _build_df(self) -> pd.DataFrame:
        """Generates a single dataframe for all input data.
        Data is separated by provided field names for easier sampling.
        catalogue_preprocessing is called on the data here.

        Returns:
            pd.DataFrame: Single processed data frame constructed through the provided catalogue paths and field names (i.e. catalogue / imgae pair names).
        """
        df = []
        for catalogue_path, field in zip(self.catalogue_paths, self.field_names):
            tmp = self.open_catalogue(path=catalogue_path)
            tmp["field"] = field
            df.append(tmp)
        df = pd.concat(df, ignore_index=True)
        if self.catalogue_preprocessing is not None:
            df = self.catalogue_preprocessing(df)
        return df

    def _build_images(
        self, drop_axes: List[int]
    ) -> Tuple[Dict[Union[str, int], Any], Dict[Union[str, int], Any]]:
        """Reads in the fields. Returns a dict of arrays and a dict of wcs coordinates.

        Args:
            drop_axes (List[int]): Not implemented.

        Returns:
            Tuple[Dict[Union[str, int], Any], Dict[Union[str, int], Any]]: Dict of arrays and a dict of coordinates (wcs). One entry each for the respective provided field names.
        """
        if self.spectral_axis:
            cubes, wcs = {}, {}
            for image_path, field in zip(self.image_paths, self.field_names):
                if self.stokes_axis:
                    cube = StokesSpectralCube.read(image_path)
                else:
                    cube = SpectralCube.read(image_path)
                # https://spectral-cube.readthedocs.io/en/latest/accessing.html#data-values
                cubes[field] = cube
                wcs[field] = cube.wcs

            return cubes, wcs
        else:
            images, wcs = {}, {}
            for image_path, field in zip(self.image_paths, self.field_names):
                data, wcs_ = self.open_fits(
                    path=image_path,
                    index=self.fits_index_images,
                )

                images[field] = data
                wcs[field] = wcs_
            for field in self.field_names:
                if self.wcs_preprocessing is not None:
                    wcs[field] = self.wcs_preprocessing(wcs[field], field)
            return images, wcs

    def open_fits(self, path: str, index: int, drop_axes: List[int] = None) -> tuple:
        """Opens fits data.

        Args:
            path (str): Path to data.
            index (int): Index at which the data product is stored within the fits file.
            drop_axes (List[int], optional): Not Implemented. Defaults to None.

        Returns:
            tuple: _description_
        """
        with fits.open(path, memmap=self.memmap) as hdul:
            data = hdul[index].data
            wcs = WCS(hdul[index].header, naxis=2)
        return data, wcs

    def open_catalogue(
        self, path: str, pandas: bool = True
    ) -> Union[Table, pd.DataFrame]:
        """Opens a catalogue either as pandas dataframe or astropy Table object.

        Args:
            path (str): Path to file.
            pandas (bool, optional): Return as pandas dataframe. Defaults to True.
            format (str, optional): File format passed to astropy table.read() call. Defaults to "fits".

        Returns:
            Union[Table, pd.DataFrame]: _description_
        """
        _path = Path(path)
        if _path.is_file():
            if _path.suffix == ".fits":
                table = Table.read(path, memmap=True, format="fits")
            elif _path.suffix == ".txt":
                table = Table.read(path, format="ascii.commented_header")
            else:
                raise ValueError("Catalogue format not recognised")
        else:
            raise ValueError("The path parameter is not a file.")

        if pandas:
            return table.to_pandas()
        else:
            return table

    def _paths_exist(self, paths: List[str]) -> List[bool]:
        """Check if paths exist.

        Args:
            paths (List[str]): List of paths.

        Returns:
            List[bool]
        """
        return [os.path.exists(path) for path in paths]

    def _check_exists(self) -> None:
        """Check if data exists.

        Raises:
            ValueError: Data not found and logs which entries are not found through ordered list of booleans.
        """
        catalogues_exist = self._paths_exist(self.catalogue_paths)
        images_exist = self._paths_exist(self.image_paths)
        if all(catalogues_exist) and all(images_exist):
            return
        else:
            raise ValueError(
                f"Data was not found at given paths:\n\tCatalogues Exist: {catalogues_exist}\n\tImages Exist: {images_exist}"
            )

    def _verify_input_lengths(self) -> None:
        """Check that catalogues, images, fits_index_catalogues, fits_index_images,
        fields all have correct lengths in relation to one another."""
        if (
            len(self.image_paths)
            != len(self.catalogue_paths)
            != len(self.field_names)
            != len(self.image_paths)
        ):
            raise ValueError(
                f"""Paths and fields must have same number of entries. Currently there are {len(self.image_paths)} image_paths, {len(self.catalogue_paths)} catalogue_paths, {len(self.field_names)} field_names"""
            )
        return
