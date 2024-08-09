from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
import numpy as np
import pandas as pd
import os

def load_catalogue(catalogue_path):
    """Load the FITS catalogue using astropy Table and convert to a pandas DataFrame."""
    table = Table.read(catalogue_path)
    df = table.to_pandas()
    return df

def get_image_wcs_and_data(image_path):
    """Get WCS information and image data."""
    with fits.open(image_path) as hdul:
        wcs = WCS(hdul[0].header)
        image_data = hdul[0].data
    return wcs, image_data

def save_sub_catalogue(df, output_path, overwrite=True):
    """Save the sub-catalogue from a pandas DataFrame to a new FITS file using astropy Table."""
    table = Table.from_pandas(df)
    table.write(output_path, format='fits', overwrite=overwrite)

def filter_objects_by_central_pixel(df, wcs, image_data):
    """Filter objects within the RA and DEC boundaries and valid data regions using pandas."""
    df = df.dropna(subset=['RA', 'DEC'])
    
    # Convert RA and DEC to pixel coordinates
    x, y = wcs.wcs_world2pix(df['RA'].values, df['DEC'].values, 0)
    
    # Initialize mask with all True values
    valid_mask = np.ones(len(df), dtype=bool)
    
    # Check bounds
    valid_mask &= (x >= 0) & (x < image_data.shape[1]) & (y >= 0) & (y < image_data.shape[0])
    
    # Apply NaN check only on valid indices
    valid_indices = np.where(valid_mask)[0]
    valid_mask[valid_indices] &= ~np.isnan(image_data[y[valid_indices].astype(int), x[valid_indices].astype(int)])
    
    # Filter DataFrame
    valid_df = df[valid_mask]
    print(f"STRONG FILTER: samples: {valid_df.shape[0]}")

    return valid_df

def filter_by_mosaic_id(df, image_path):
    df = df.dropna(subset=['RA', 'DEC'])
    field_name = os.path.dirname(image_path).split('/')[-1]
    original_sample_count = df.shape[0]
    df = df.loc[df['Mosaic_ID'] == field_name.encode('UTF-8')]
    print(f"SIMPLE FILTER: field_name: {field_name}; samples: {df.shape[0]}; original_sample_count: {original_sample_count}; sample_estimate: {int(1/841*original_sample_count)}")
    return df

def main(catalogue_path, image_paths):
    catalogue = load_catalogue(catalogue_path)
    for image_path in image_paths:
        wcs, image_data = get_image_wcs_and_data(image_path)
        # Apply filters
        sub_catalogue_ = filter_by_mosaic_id(catalogue, image_path)
        sub_catalogue = filter_objects_by_central_pixel(sub_catalogue_, wcs, image_data)
        # Get parent directory of the image and create a new file with the same name as the image but with _sub_catalogue.fits
        output_path = os.path.join(os.path.dirname(image_path), os.path.basename(catalogue_path).replace('.fits', '_sub_catalogue.fits'))
        save_sub_catalogue(sub_catalogue, output_path)

if __name__=="__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--catalogue_path', help='Path to the FITS catalogue.', default='data/lotssdr2/combined-release-v1.1-LM_opt_mass.fits')
    argparser.add_argument('-i', '--image_paths', help='Comma-separated paths to the FITS images.', default='data/lotssdr2/public')

    args = argparser.parse_args()

    # Check if image_paths is a directory 
    if os.path.isdir(args.image_paths):
        # Recurse through the directory to get all the fits files
        image_paths = []
        for root, dirs, files in os.walk(args.image_paths):
            for file in files:
                if file.endswith('mosaic-blanked.fits'):
                    image_paths.append(os.path.join(root, file))
    else:
        image_paths = args.image_paths.split(',')
    print(image_paths)
    main(args.catalogue_path, image_paths)
