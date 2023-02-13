import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.units import Quantity
from typing import Any, Callable, Optional, List, Union, Tuple, Dict
from astropy.nddata import Cutout2D
import os


class CataData:
    """A class taking fits catalogues and images and producing
    a data set for deep learning."""

    def __init__(
        self,
        catalogue_paths: Union[List[str], str],
        image_paths: Union[List[str], str],
        field_names: Union[List[Union[int, int]], str],
        cutout_width: int | Quantity = 32,
        memmap: bool = False,
        polarisation: bool = False,
        transform: Optional[Callable] = None,
        catalogue_preprocessing: Optional[Callable] = None,
        image_preprocessing: Optional[Callable] = None,
        wcs_preprocessing: Optional[Callable] = None,
        fits_index_catalogue: int = 1,
        fits_index_images: int = 0,
        image_drop_axes: List[int] = [3, 2],
        origin: int = 1,
    ) -> None:
        """Produces a deep learning ready data set from fits catalogues
        and fits images.

        Args:
            catalogue_paths (list[str] | str):
                Fits catalogue path(s) in matching order.
            image_paths (list[str] | str):
                Fits image path(s) in matching order.
            field_names (list[str  |  int] | str):
                Names of the field(s) in matching order.
            cutout_width (int, optional):
                Cut out pixel width. Defaults to 32.
            memmap (bool, optional):
                Whether to use memory mapping (dynamic reading of images into memory). Defaults to False.
            polarisation (bool, optional):
                Whether data is polarised or not. Defaults to False.
            transform (Optional[Callable], optional):
                Transformations to use. Currently not implemented. Defaults to None.
            catalogue_preprocessing (Optional[Callable], optional):
                Function to apply to catalogues before use. Ideal for filtering to subsamples. Defaults to None.
            image_preprocessing (Optional[Callable], optional):
                Function to apply to images before use. Defaults to None.
            wcs_preprocessing (Optional[Callable], optional):
                Function applied to the astropy wcs object before selecting data from the images. Defaults to None.
            fits_index_catalogue (int, optional): Index used in self.open_fits call. Selects correct wcs entry for respective catalogues. Ordered appropriately with paths. Defaults to 1.
            fits_index_images (int, optional): Index used in self.open_fits call. Selects correct wcs entry for respective images. Ordered appropriately with paths. Defaults to 0.
            image_drop_axes (list[int], optional): Not Implemented. Defaults to [3,2].
            origin (int, optional): Wcs origin. Used in cutout to calculated wcs.world2pix coords. Defaults to 1.
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
        self.image_preprocessing = image_preprocessing
        self.wcs_preprocessing = wcs_preprocessing

        # if transform is albumentations transform do ...
        self.transform = transform

        self.memmap = memmap
        self.origin = origin
        self.polarisation = polarisation
        self.cutout_width = cutout_width

        self.df = self._build_df()
        self.images, self.wcs = self._build_images(image_drop_axes)

    def __getitem__(self, index: int, return_wcs=False) -> np.ndarray:
        if index >= self.__len__():
            raise ValueError(f"Index out of range.")
        coords = self.df.iloc[index : index + 1][["RA", "DEC"]].values
        field = self.df.iloc[index].field
        image = self.cutout(coords, field=field)[0]
        return self.cutout(coords, field=field, return_wcs=return_wcs)

    def __len__(self) -> int:
        return len(self.df)

    def cutout(
        self, coords: np.ndarray, field: Union[str, int], return_wcs: bool = False
    ) -> Union[tuple, np.ndarray]:
        """Produces a set of images based on the provided
        coordinates and field.
        """
        ### Currently using:
        # https://docs.astropy.org/en/stable/nddata/utils.html#cutout-images
        # https://docs.astropy.org/en/stable/api/astropy.nddata.Cutout2D.html
        positions = self.wcs[field].all_world2pix(coords, self.origin)
        cutouts = []
        wcs = []
        for x, y in positions:
            cutout = Cutout2D(
                self.images[field],
                (x, y),
                (self.cutout_width, self.cutout_width),
                wcs=self.wcs[field],
                mode="partial",
                fill_value=0,
            )
            if return_wcs:
                wcs.append(cutout.wcs)
            cutouts.append(cutout.data)
        if return_wcs:
            return np.stack(cutouts), wcs
        return np.stack(cutouts)

    def save_cutout(path: str, index: int, format: str = "fits") -> None:
        """Saves the cutout of the respective index.
        Args:
            path (str): Path which the file should be saved to.
            index (int): Catalogue index used to produce cutout.
            format (str): File out type. Defaults to 'fits' (coordinate system corrected header).
        """
        cutout, wcs = self.__getitem__(index, return_wcs=True)
        wcs = wcs[0]  # unpack extract dimension
        cutout = cutout[0]
        if format == "fits":
            fits.writeto(path, data=cutout, header=wcs.to_header())
        else:
            raise NotImplementedError("Currently only fits format supported.")
        return

    def plot(self, index: int) -> None:
        """Plot the source with the given idx."""
        crosshair_alpha = 0.3
        image, wcs = self.__getitem__(index, return_wcs=True)
        image = np.squeeze(image[0])
        wcs = wcs[0]
        plt.subplot(projection=wcs)
        plt.imshow(image, origin="lower", cmap="Greys")
        plt.colorbar()
        plt.contour(
            image,
            levels=[self.df.iloc[index]["ISL_RMS"] * (3 + n) for n in range(3)],
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
        """
        df = []
        for catalogue_path, field in zip(self.catalogue_paths, self.field_names):
            data, wcs = self.open_fits(
                path=catalogue_path, index=self.fits_index_catalogue
            )
            tmp = pd.DataFrame(data)
            tmp["field"] = field
            df.append(tmp)
        df = pd.concat(df, ignore_index=True)
        if self.catalogue_preprocessing is not None:
            df = self.catalogue_preprocessing(df)
        return df

    def _build_images(
        self, drop_axes: List[int]
    ) -> Tuple[Dict[Union[str, int], np.ndarray], Dict[Union[str, int], any]]:
        """Reads in the fields. Returns a dict of arrays and a dict of wcs coordinates."""
        images, wcs = {}, {}
        for image_path, field in zip(self.image_paths, self.field_names):
            data, wcs_ = self.open_fits(
                path=image_path,
                index=self.fits_index_images,
                # drop_axes=drop_axes
            )
            images[field] = data
            wcs[field] = wcs_
        if self.wcs_preprocessing is not None:
            for field, wcs_ in wcs.items():
                wcs[field] = self.wcs_preprocessing(wcs_, field)
        if self.image_preprocessing is not None:
            for field, image in images.items():
                images[field] = self.image_preprocessing(image, field)
        return images, wcs

    def open_fits(self, path: str, index: int, drop_axes: List[int] = None) -> tuple:
        with fits.open(path, memmap=self.memmap) as hdul:
            data = hdul[index].data
            wcs = WCS(hdul[index].header, naxis=2)
        return data, wcs

    def _paths_exist(self, paths: List[str]) -> List[bool]:
        return [os.path.exists(path) for path in paths]

    def _check_exists(self) -> Tuple[pd.DataFrame, List[np.ndarray]]:
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
        fields all have correct lengths in relation to one another.
        """
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
