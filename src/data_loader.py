"""
Data loading utilities for the "To bee or not to bee" project.

Handles reading images, masks, and the Excel classification file, and
building a feature DataFrame for the whole training set.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from features import extract_all_features


def load_image(path):
    """Load an RGB image as a numpy uint8 array of shape (H, W, 3)."""
    img = Image.open(path).convert('RGB')
    return np.array(img)


def load_mask(path):
    """Load a binary mask as a numpy uint8 array of shape (H, W)."""
    m = Image.open(path).convert('L')
    return (np.array(m) > 127).astype(np.uint8)


def find_pair(img_dir, mask_dir, image_id):
    """
    Locate the image and mask files for a given id.
    Convention : images = `<id>.JPG`, masks = `binary_<id>.tif`
    Robust fallback for other conventions too.
    """
    img_dir = Path(img_dir)
    mask_dir = Path(mask_dir)

    # ID can be an int from Excel — convert to string variants
    n = int(image_id)
    id_variants = [str(n), f'{n:03d}', f'{n:04d}']

    image_exts = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG',
                  '.tif', '.tiff', '.TIF', '.TIFF', '.bmp')

    def search(directory, prefixes, suffixes):
        for v in id_variants:
            for prefix in prefixes:
                for suffix in suffixes:
                    for ext in image_exts:
                        p = directory / f'{prefix}{v}{suffix}{ext}'
                        if p.exists():
                            return p
        # Fallback: scan directory and match the id as a stand-alone number
        import re
        if not directory.exists():
            return None
        for p in directory.iterdir():
            if not p.is_file() or p.suffix.lower() not in [e.lower() for e in image_exts]:
                continue
            stem = p.stem
            for v in id_variants:
                if re.search(rf'(^|[^0-9]){re.escape(v)}([^0-9]|$)', stem):
                    return p
        return None

    img_path = search(img_dir,
                      prefixes=['', 'image_', 'img_', 'IMG_'],
                      suffixes=[''])
    mask_path = search(mask_dir,
                       prefixes=['binary_', '', 'mask_', 'Mask_'],
                       suffixes=['', '_mask', '_binary'])

    return img_path, mask_path


def build_feature_dataframe(labels_df, img_dir, mask_dir, id_column='ID',
                            show_progress=True):
    """
    Iterate through the labels DataFrame, extract features for every (image, mask)
    pair, and merge with the labels.

    Parameters
    ----------
    labels_df : pandas.DataFrame
        Must contain an 'ID' column.
    img_dir, mask_dir : str or Path
    id_column : str

    Returns
    -------
    pandas.DataFrame indexed by ID with features + labels.
    """
    rows = []
    iterator = labels_df.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(labels_df), desc='Extracting features')

    for _, row in iterator:
        image_id = row[id_column]
        img_path, mask_path = find_pair(img_dir, mask_dir, image_id)
        if img_path is None or mask_path is None:
            print(f'[warning] missing files for id={image_id} '
                  f'(img={img_path}, mask={mask_path})')
            continue
        image = load_image(img_path)
        mask = load_mask(mask_path)
        # Some datasets store masks at a different resolution: resize if needed
        if mask.shape != image.shape[:2]:
            from PIL import Image as PILImage
            mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((image.shape[1], image.shape[0]),
                                       resample=PILImage.NEAREST)
            mask = (np.array(mask_pil) > 127).astype(np.uint8)

        feats = extract_all_features(image, mask)
        feats[id_column] = image_id
        rows.append(feats)

    feat_df = pd.DataFrame(rows)
    merged = labels_df.merge(feat_df, on=id_column, how='inner')
    return merged
