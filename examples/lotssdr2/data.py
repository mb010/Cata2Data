import pandas as pd
import os
import numpy as np
import cata2data
import torch

from typing import Optional

# Image Preprocessing
class CustomTransform(torch.nn.Module):
    def __init__(self, transformations = [], range_normalize = False):
        super(CustomTransform, self).__init__()
        self.transformations = transformations
        self.range_normalize = range_normalize

        # define asinh transformation
        def asinh_trans(x) :
            asinh_x = torch.asinh(x)
            return asinh_x

        # define tanh transformation
        def tanh_trans(x) :
            return 0.5 * (1.0 + torch.tanh(x))

        # define log transformation
        def log_trans(x) :
            # log_x = torch.log1p(torch.where(x < 0, 0, x)) # Can try this if the below still explodes / nans.
            log_x = torch.log1p(x)
            return log_x
        
        # define linear transformation
        def linear_trans(x) :
            return (x - x.min()) / (x.max() - x.min())
        
        # define random transformation
        def rand_trans(x) :
            rand_x = torch.rand_like(x, dtype = torch.float)
            return rand_x

        # define clamping
        def cust_clamp(x) :
            return torch.clamp(x, min = 0, max = 1)

        # define function for removing possible subnormal numbers
        def rem_tiny(x) :
            tinies = (np.abs(x) <= .00001)
            x[tinies] = .00001
            return x

        # define rounding
        def cust_round(x) :
            return torch.round(x * 10000) / 10000.0
        
        # define jitter
        def cust_jitter(x) :
            return x + torch.randn(x.size(), device = x.device, dtype = x.dtype) * .1

        # define alternate way to random
        def rand_trans2(x) :
            random_vals = torch.rand(x.shape)
            x[:] = random_vals
            return x

        # define function to make sure data is all finite
        def make_finite(x) :
            non_fin = ~torch.isfinite(x)
            x[non_fin] = 0
            return x

        # define no transformation
        def no_trans(x) :
            return x

        # define transformation mappings
        self.transformation_map = {
            "asinh": asinh_trans,
            "tanh": tanh_trans,
            "log": log_trans,
            "linear": linear_trans,
            "random": rand_trans,
            "clamp": cust_clamp,
            "round": cust_round,
            "jitter": cust_jitter,
            "random2": rand_trans2,
            "tiny": rem_tiny,
            "finite": make_finite,
            "none": no_trans
        }

    def forward(self, img):
        for trans in self.transformations:
            if trans in self.transformation_map:
                img = self.transformation_map[trans](img)
            else:
                raise ValueError(f"Unknown transformation: {trans}")
        if self.range_normalize:
            img = self.transformation_map['linear'](img)
        return img

class PermuteAxes(torch.nn.Module):
    def __init__(self, axes=(1, 0, 2)):
        super(PermuteAxes, self).__init__()
        self.axes = axes
    def forward(self, img):
        img = torch.permute(img, self.axes)
        return img

class LoTTS_Preprocessing():
    def __init__(self, patch_size: Optional[int]=None, scaling: float = 1.2, square: bool=False, cutout_size_name: list=["cutout_height", "cutout_width"]):
        self.patch_size = patch_size
        self.scaling = scaling
        self.square = square
        self.cutout_size_name = cutout_size_name
    
    def set_ceiling(self, x):
        return int(np.ceil(x/self.patch_size)*self.patch_size)
    
    def __call__(self, df):
        """Preprocess the LoTTS catalogue. Stages:
        0. Drop all columns not used
        1. Calculate cutout size from composite measurements.
        2. Calculate cutout size from source finder measurements.
        3. Combine sizes (composite measurements take precedence over source finder measurements).
        4. Drop rows where cutout size is nan
        5. Ensure size is a multiple of patch_size (and minimum of patch_size) if provided.
        6. Select correct right ascension and declination (optical if available, else radio).
        7. Adjust Isl_rms to mJy
        8. Select only S_Code [M, Z, C] multicomponent sources. (S = single, M = multi, C = complex, Z = zoo)
        9. Set Total_flux to np.float32
        10. Set log10_Total_flux to log10(Total_flux)
        11. Drop all columns not needed
        12. Drop all samples where patch length would be longer than 512

        Args:
            df (pd.DataFrame): LoTTS catalogue

        Returns:
            pd.DataFrame
        """

        # Drop all columns not used
        df = df[["field", "Source_Name", "RA", "DEC", "optRA", "optDec", "Isl_rms", "Total_flux", "Maj", "Min", "PA", "Composite_Width", "Composite_Size", "Composite_PA", "S_Code"]]

        df = df.drop_duplicates(subset=["Source_Name"], keep="first")

        # Initialize cutout size columns with nan
        df[[*self.cutout_size_name]] = pd.DataFrame([[None, None]]*len(df), index=df.index)

        # Get sizes from measurements (composite measurements)
        df.loc[:, "Composite_PA"] += 90
        composite_sizes = cata2data.preprocessing.calculate_cutout_size(df, scaling=self.scaling, square=self.square, cutout_size_name=self.cutout_size_name, major_axis="Composite_Width", minor_axis="Composite_Size", angle="Composite_PA")

        # Get sizes from measurements
        # df.loc[:, "PA"] = df.loc[:, "PA"].apply(lambda x: x+90) # TODO: Check if this is still true. Adding 90 degrees to the DC_PA to match the composite measurements
        df.loc[:, "PA"] += 90
        dc_sizes = cata2data.preprocessing.calculate_cutout_size(df, scaling=self.scaling, square=self.square, major_axis="Maj", minor_axis="Min", angle="PA")

        # Combine sizes
        sizes = composite_sizes.combine_first(dc_sizes)
        df.loc[sizes.index, self.cutout_size_name] = sizes
        
        # Drop rows where cutout size is nan
        df = df.dropna(subset=self.cutout_size_name)
        for col in self.cutout_size_name:
            df.loc[:, col] = df[col].astype(float)
        
        # Ensure size is a multiple of patch_size (and minimum of patch_size)
        if self.patch_size:
            df.loc[:, "cutout_height"] = df["cutout_height"].apply(self.set_ceiling)
            df.loc[:, "cutout_width"] = df["cutout_width"].apply(self.set_ceiling)

        # Select correct right ascension and declination
        df.rename(columns={"optRA": "ra", "optDec": "dec"}, inplace=True)       
        # Where ra, dec are nan fill with RA DEC (radio ra and dec)
        df.loc[:, "ra"] = df.loc[:, "ra"].fillna(df.loc[:, "RA"])
        df.loc[:, "dec"] = df.loc[:, "dec"].fillna(df.loc[:, "DEC"])

        # Adjust Isl_rms to mJy
        df.loc[:, "Isl_rms"] = df.loc[:, "Isl_rms"]*0.001
        df = df[df["S_Code"] != b"S"]
        df.loc[:, "Total_flux"] = df.loc[:, "Total_flux"].astype(np.float32)
        df.loc[:, "log10_Total_flux"] = np.log10(df.loc[:, "Total_flux"])

        # Drop all columns not used
        df = df[["field", "Source_Name", "ra", "dec", "Isl_rms", "log10_Total_flux", "Total_flux", "cutout_height", "cutout_width", "S_Code"]]
        
        # Drop all samples where patch length would be longer than 512
        df = df[df["cutout_height"] * df["cutout_width"] / self.patch_size**2 <= 512]
        
        return df


def LoTTSDataset(
        data_folder,
        catalogue_name="combined-release-v1.1-LM_opt_mass_sub_catalogue.fits",
        image_name="mosaic-blanked.fits",
        square=True,
        patch_size=16,
        cutout_scaling=1.2,
        transform=None,
        targets="log10_Total_flux",
        cutout_shape=("cutout_width", "cutout_height"),
        sample_limit=None,
        train=True,
        train_size=0.8,
    ) -> cata2data.CataData:

    # Check if image_paths is a directory 
    # Recurse through the directory to get all the fits files
    image_paths = []
    catalogue_paths = []
    field_names = []
    for root, dirs, files in os.walk(data_folder):
        named = False
        for file in files:
            if file.endswith(image_name):
                image_paths.append(os.path.join(root, file))
            elif file.endswith(catalogue_name):
                catalogue_paths.append(os.path.join(root, file))
            if not named:
                # Append parent dir name
                field_names.append(os.path.basename(root))
                named = True
    
    # Define preprocessing
    catalog_preprocessing = LoTTS_Preprocessing(
        patch_size=patch_size, 
        scaling=cutout_scaling, 
        square=square, 
        cutout_size_name=cutout_shape
    )

    print(f"Found {len(image_paths)} images and {len(catalogue_paths)} catalogues in {data_folder}. Using {len(field_names)} fields.")

    # Determine the start and end index of the images
    if train:
        start = 0
        end = int(len(image_paths)*train_size)
        if end == start:
            end = start + 1
            print("WARNING: Only one image found. Using this image for training. Consider training on Sample size instead.")
    else:
        start = int(len(image_paths)*train_size)
        end = len(image_paths)

    if sample_limit:
        end = start + sample_limit
    
    # Create the dataset
    data = cata2data.CataData(
        catalogue_paths=catalogue_paths[start:end],
        image_paths=image_paths[start:end],
        field_names=field_names[start:end],
        cutout_shape=cutout_shape,
        catalogue_preprocessing=catalog_preprocessing,
        targets = targets,
        transform=transform,
    )

    # Return the dataset
    return data
