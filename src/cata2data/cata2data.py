import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence

import astropy.units as units
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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
        catalogue_paths: Union[List[str], Optional[str]],
        image_paths: Union[List[str], str],
        field_names: Union[List[int], List[str], str],
        cutout_shape: Union[int, Sequence[Union[int, str]]] = (32, 32),
        memmap: bool = True,
        targets: Union[Optional[str], List[str]] = None,
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
        fill_value: float = 0.0,
        overlap: float = 16,
        **kwargs,
    ) -> None:
        """Produces a deep learning ready data set from fits catalogues
        and fits images.

        Args:
            catalogue_paths (List[str] | Optional[str]):
                Fits catalogue path(s) in matching order.
            image_paths (List[str] | str):
                Fits image path(s) in matching order.
            field_names (List[str  |  int] | str):
                Names of the field(s) in matching order.
            cutout_shape (Union[int, Sequence[int], Sequence[str]], optional):
                Shape of the cutout. Defaults to (32, 32). If strings are provided, these are used as keys to the catalogue to extract the shape for each entry.
            memmap (bool, optional):
                Whether to use memory mapping (dynamic reading of images into memory). Defaults to False.
            targets (bool, optional):
                Column names of the targets in the catalogue. If it is a string, it is converted to a list of length 1. Defaults to None.
            transform (Optional[Callable], optional):
                Transformations to use. Defaults to None. Using torchvision.transforms.v2.ToImage() is recommended.
            catalogue_preprocessing (Optional[Callable], optional):
                Function to apply to catalogues before use. Ideal for filtering to subsamples. Defaults to None.
            wcs_preprocessing (Optional[Callable], optional):
                Function applied to the astropy wcs object before selecting data from the images. Defaults to None.
            fits_index_catalogue (int, optional):
                Index used in self.open_fits call. Selects correct wcs entry for
                respective catalogues. Ordered appropriately with paths. Defaults to 1.
            fits_index_images (int, optional):
                Index used in self.open_fits call. Selects correct wcs entry for
                respective images. Ordered appropriately with paths. Defaults to 0.
            image_drop_axes (List[int], optional):
                Not Implemented. Defaults to [3,2].
            origin (int, optional):
                Wcs origin. Used in cutout to calculated wcs.world2pix coords. Defaults to 1.
            spectral_axis (bool):
                TO BE COMPLETED. Detaults to False.
            stokes_axis (bool):
                TO BE COMPLETED. Defaults to False.
            fill_value (float):
                Value which cutouts should be padded with if there isn't full coverage. Defaults to 0.

        """
        if catalogue_paths is not None:
            self.catalogue_paths = (
                catalogue_paths if type(catalogue_paths) is list else [catalogue_paths]
            )
        else:
            self.catalogue_paths = catalogue_paths
        self.image_paths = image_paths if type(image_paths) is list else [image_paths]
        self.field_names = field_names if type(field_names) is list else [field_names]
        self.fits_index_catalogue = fits_index_catalogue
        self.fits_index_images = fits_index_images

        if targets is not None:
            self.targets = targets if isinstance(targets, list) else [targets]
        else:
            self.targets = targets

        # Checks
        self._verify_input_lengths()
        self._check_exists()

        self.catalogue_preprocessing = catalogue_preprocessing
        self.wcs_preprocessing = wcs_preprocessing
        self.transform = transform
        self.kwargs = kwargs

        self.memmap = memmap
        self.origin = origin
        if isinstance(cutout_shape, int) or isinstance(cutout_shape, float):
            cutout_shape = (int(cutout_shape), int(cutout_shape))
        self.cutout_width, self.cutout_height = cutout_shape
        self.spectral_axis = spectral_axis
        self.stokes_axis = stokes_axis
        self.return_wcs = return_wcs

        self.fill_value = fill_value
        self.images, self.wcs = self._build_images(image_drop_axes)
        self.df = self._build_df() if catalogue_paths is not None else self._synthesise_df(overlap=overlap)

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

        height, width = self.__get_cutout_size__(index)
        if return_wcs:
            img, wcs_ = self.cutout(coords, field=field, height=height, width=width, return_wcs=return_wcs)
        else:
            img = self.cutout(coords, field=field, height=height, width=width, return_wcs=return_wcs)
        if self.transform:
            # NOTE TODO: This should hopefully be a temporary fix so the axes are in the expected order.
            img = np.transpose(img, (1, 2, 0)) # (C, H, W) -> (H, W, C) which is the expected order for ndarrays
            img = self.transform(img)
        if self.targets:
            targets = self.df.iloc[index][self.targets].values

        output = (img,)
        if self.targets:
            output += (targets,)
        if return_wcs:
            output += (wcs_,)
        return output

    def __len__(self) -> int:
        """Returns the length of the processed catalogue. Necessary for pytorch dataloaders.

        Returns:
            int: Data set length.
        """
        return len(self.df)
    
    def __get_cutout_size__(self, index: int) -> tuple:
        """Returns the size of the cutout at the given index.

        Args:
            index (int): Index of the cutout.

        Returns:
            tuple: Size of the cutout.
        """
        size = []
        for dimension in [self.cutout_height, self.cutout_width]:
            if isinstance(dimension, str):
                if dimension not in self.df.columns:
                    raise ValueError(f"Column '{dimension}' not found in catalogue.")
                size.append(self.df.iloc[index][dimension])
            # If numeric
            elif isinstance(dimension, (int, float)):
                size.append(int(dimension))
            else:
                raise ValueError("Cutout dimensions must be strings or numeric values.")
        height, width = size
        return height, width

    def cutout(
        self, coords: np.ndarray, field: Union[str, int], height: int, width: int, return_wcs: bool = False,
    ) -> Union[tuple, np.ndarray]:
        """Produces a set of images based on the provided
        coordinates and field.

        Args:
            coords (np.ndarray): Source coordinates (ra, dec).
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
                    height,
                    width,
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
                        height,
                        width,
                    ),
                    wcs=self.wcs[field],
                    mode="partial",
                    fill_value=self.fill_value,
                )
                wcs = cutout.wcs
                cutouts.append(cutout.data)
                if return_wcs:
                    wcs_.append(cutout.wcs)

        if "image_preprocessing" in self.kwargs.keys():
            cutouts = self.kwargs["image_preprocessing"](cutouts)

        cutouts = np.stack(cutouts)
        if return_wcs:
            return cutouts, wcs_
        return cutouts

    def save_cutout(self, path: str, index: int, format: str = "fits") -> None:
        """Saves the cutout of the respective index.
        Args:
            path (str): Path which the file should be saved to.
            index (int): Catalogue index used to produce cutout.
            format (str): File out type. Defaults to 'fits' (coordinate system corrected header).
        """
        if self.spectral_axis:
            raise NotImplementedError
            
        if self.targets:
            cutout, _, wcs = self.__getitem__(index, force_return_wcs=True)
        else:
            cutout, wcs = self.__getitem__(index, force_return_wcs=True)
            
        wcs = wcs[0]  # unpack extract dimension
        cutout = cutout[0]
        if format == "fits":
            fits.writeto(path, data=cutout, header=wcs.to_header())
        else:
            raise NotImplementedError("Currently only fits format supported.")
        return

    def plot(
        self, index: int, contours: bool = False, sigma_name: str = "ISL_RMS", min_sigma: int = 3, log_scaling: bool = False, title: str = None,
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
        out = self.__getitem__(index, force_return_wcs=True)
        image, wcs = (out[0], out[2]) if self.targets else out

        #image = np.squeeze(image[0])
        image = np.squeeze(image)
        wcs = wcs[0]
        
        height, width = self.__get_cutout_size__(index)
        plt.subplot(projection=wcs)
        plt.imshow(image, origin="lower", cmap="Greys",  norm=colors.LogNorm() if log_scaling else None)
        if title:
            plt.title(title)
        plt.colorbar()
        if contours:
            plt.contour(
                image,
                levels=[self.df.iloc[index][sigma_name] * (min_sigma + n) for n in range(3)],
                origin="lower",
            )
        plt.axhline(height // 2, color="red", linewidth=2, ls="--", alpha=crosshair_alpha)
        plt.axvline(width // 2, color="red", linewidth=2, ls="--", alpha=crosshair_alpha)
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

    def _synthesise_df(self, overlap: float, **kwargs):
        """Generate a df that samples the images as patches across the frame.

        Args:
            overlap (float): Fractional overlap between patches or stride size if greater than 1.
        
        Returns:
            pd.DataFrame: Data frame of ra, dec and field.
        """
        df = {'ra': [], 'dec': [], 'field': []}
        # For each iamge
        if overlap<1.0:
            overlap_width  = int(self.cutout_width*overlap)
            overlap_height = int(self.cutout_height*overlap)
        else:
            overlap_width  = overlap 
            overlap_height = overlap
        stride = (self.cutout_width-overlap_width, self.cutout_height-overlap_height)

        if stride[0] < 1 or stride[1] < 1:
            raise ValueError(f"Overlap is too large for cutout size. Stride must be greater than 0. Current stride: {stride} with overlap: {overlap} and cutout size: {self.cutout_width}x{self.cutout_height}")
        if stride[0] > self.cutout_width or stride[1] > self.cutout_height:
            raise Warning(f"Overlap is too large for cutout size. Stride must be smaller than the cutout size. Current stride: {stride} with overlap: {overlap} and cutout size: {self.cutout_width}x{self.cutout_height}")
        
        # Stride across the image and calculate ra+dec for the cutout to store in the table
        for field in self.field_names:
            wcs = self.wcs[field]
            image = self.images[field]
            x_start = self.cutout_width//2
            y_start = self.cutout_height//2
            x = [x_start]
            y = [y_start]
            df['field'] += [field]
            for i in range((image.shape[-1]-self.cutout_width)//stride[0]):
                for j in range((image.shape[-2]-self.cutout_height)//stride[1]):
                    x.append(x_start+i*stride[0])
                    y.append(y_start+j*stride[1])
                    df['field'] += [field]
            # Get ra and dec using wcs
            ra, dec = wcs.all_pix2world(y, x, self.origin)
            df['ra'] += list(ra)
            df['dec'] += list(dec)
        df = pd.DataFrame(df).dropna()
        return df

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
        catalogues_exist = self._paths_exist(self.catalogue_paths) if self.catalogue_paths is not None else [True]
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
        if self.catalogue_paths is None:
            if (
                len(self.image_paths)
                != len(self.field_names)
            ):
                raise ValueError(
                    f"""Paths and fields must have same number of entries. Currently there are {len(self.image_paths)} image_paths, {len(self.field_names)} field_names. No catalogues provided."""
                )
            return
        if (
            len(self.image_paths)
            != len(self.catalogue_paths)
            != len(self.field_names)
        ):
            raise ValueError(
                f"""Paths and fields must have same number of entries. Currently there are {len(self.image_paths)} image_paths, {len(self.catalogue_paths)} catalogue_paths, {len(self.field_names)} field_names"""
            )
        return
