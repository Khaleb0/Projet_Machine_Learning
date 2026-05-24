"""
Feature extraction module for the "To bee or not to bee" project.

Extracts features from insect images using their segmentation masks.
All features required by the project specification (III.1) are implemented,
plus several additional features.
"""

import numpy as np
from skimage import measure, color
from skimage.feature import graycomatrix, graycoprops
from scipy import ndimage


# -----------------------------------------------------------------------------
# Required features (III.1)
# -----------------------------------------------------------------------------

def bug_pixel_ratio(mask):
    """Ratio of bug pixels to total image pixels."""
    mask_bin = (mask > 0).astype(np.uint8)
    return mask_bin.sum() / mask_bin.size


def rgb_stats_in_mask(image, mask):
    """
    Min, max, mean, median and std for R, G, B channels inside the bug mask.

    Returns
    -------
    dict with keys r_min, r_max, r_mean, r_median, r_std,
                  g_min, g_max, g_mean, g_median, g_std,
                  b_min, b_max, b_mean, b_median, b_std
    """
    mask_bool = mask > 0
    stats = {}
    channels = ['r', 'g', 'b']
    for i, ch in enumerate(channels):
        pixels = image[..., i][mask_bool]
        if pixels.size == 0:
            stats[f'{ch}_min'] = 0
            stats[f'{ch}_max'] = 0
            stats[f'{ch}_mean'] = 0
            stats[f'{ch}_median'] = 0
            stats[f'{ch}_std'] = 0
        else:
            stats[f'{ch}_min'] = float(pixels.min())
            stats[f'{ch}_max'] = float(pixels.max())
            stats[f'{ch}_mean'] = float(pixels.mean())
            stats[f'{ch}_median'] = float(np.median(pixels))
            stats[f'{ch}_std'] = float(pixels.std())
    return stats


def shape_symmetry(mask):
    """
    Shape symmetry measure based on the bug mask.

    We measure horizontal and vertical symmetry of the bounding box of the mask:
    a perfectly symmetric shape yields 1, a totally asymmetric one yields 0.
    """
    mask_bin = (mask > 0).astype(np.uint8)
    if mask_bin.sum() == 0:
        return {'shape_sym_h': 0.0, 'shape_sym_v': 0.0}

    # Crop to bounding box
    ys, xs = np.where(mask_bin > 0)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = mask_bin[y0:y1, x0:x1]

    # Horizontal symmetry (left-right flip)
    flipped_h = np.fliplr(crop)
    sym_h = 1 - np.abs(crop.astype(float) - flipped_h.astype(float)).mean()

    # Vertical symmetry (up-down flip)
    flipped_v = np.flipud(crop)
    sym_v = 1 - np.abs(crop.astype(float) - flipped_v.astype(float)).mean()

    return {'shape_sym_h': float(sym_h), 'shape_sym_v': float(sym_v)}


def color_symmetry(image, mask):
    """
    Color symmetry inside the bug mask.

    For each channel we compare the left/right and top/bottom halves of the
    bounding box restricted to the mask, returning 1 - mean absolute difference.
    """
    mask_bin = (mask > 0).astype(np.uint8)
    if mask_bin.sum() == 0:
        return {'color_sym_h': 0.0, 'color_sym_v': 0.0}

    ys, xs = np.where(mask_bin > 0)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    img_crop = image[y0:y1, x0:x1].astype(float)
    mask_crop = mask_bin[y0:y1, x0:x1]

    # Apply mask: pixels outside the bug become 0
    img_masked = img_crop * mask_crop[..., None]

    flipped_h = np.fliplr(img_masked)
    flipped_v = np.flipud(img_masked)

    # Normalize by 255 so the symmetry score is in [0, 1]
    sym_h = 1 - (np.abs(img_masked - flipped_h).mean() / 255.0)
    sym_v = 1 - (np.abs(img_masked - flipped_v).mean() / 255.0)

    return {'color_sym_h': float(sym_h), 'color_sym_v': float(sym_v)}


# -----------------------------------------------------------------------------
# Additional features (the project asks for at least 2)
# -----------------------------------------------------------------------------

def shape_descriptors(mask):
    """
    Shape descriptors of the largest connected region in the bug mask:
    - aspect ratio of the bounding box
    - extent (area / bbox area)
    - eccentricity
    - solidity (area / convex hull area)
    - perimeter / sqrt(area) — a compactness-like measure
    """
    mask_bin = (mask > 0).astype(np.uint8)
    if mask_bin.sum() == 0:
        return {
            'aspect_ratio': 0.0, 'extent': 0.0, 'eccentricity': 0.0,
            'solidity': 0.0, 'compactness': 0.0,
        }

    labels = measure.label(mask_bin)
    regions = measure.regionprops(labels)
    if not regions:
        return {
            'aspect_ratio': 0.0, 'extent': 0.0, 'eccentricity': 0.0,
            'solidity': 0.0, 'compactness': 0.0,
        }
    region = max(regions, key=lambda r: r.area)

    minr, minc, maxr, maxc = region.bbox
    height = maxr - minr
    width = maxc - minc
    aspect_ratio = width / height if height > 0 else 0.0
    compactness = (region.perimeter / np.sqrt(region.area)) if region.area > 0 else 0.0

    return {
        'aspect_ratio': float(aspect_ratio),
        'extent': float(region.extent),
        'eccentricity': float(region.eccentricity),
        'solidity': float(region.solidity),
        'compactness': float(compactness),
    }


def hsv_stats_in_mask(image, mask):
    """
    HSV statistics inside the bug mask (mean and std).
    HSV is often more discriminative than RGB for color-based classification.
    """
    mask_bool = mask > 0
    hsv = color.rgb2hsv(image / 255.0)
    stats = {}
    for i, ch in enumerate(['h', 's', 'v']):
        pixels = hsv[..., i][mask_bool]
        if pixels.size == 0:
            stats[f'hsv_{ch}_mean'] = 0.0
            stats[f'hsv_{ch}_std'] = 0.0
        else:
            stats[f'hsv_{ch}_mean'] = float(pixels.mean())
            stats[f'hsv_{ch}_std'] = float(pixels.std())
    return stats


def texture_features(image, mask):
    """
    Gray-level co-occurrence matrix (GLCM) texture features inside the bug mask.
    Useful to distinguish hairy bumblebees from smoother bees/insects.
    """
    gray = (color.rgb2gray(image) * 255).astype(np.uint8)
    mask_bin = (mask > 0)
    if mask_bin.sum() == 0:
        return {'glcm_contrast': 0.0, 'glcm_homogeneity': 0.0,
                'glcm_energy': 0.0, 'glcm_correlation': 0.0}

    # Crop to bbox of the bug to focus the GLCM on the insect
    ys, xs = np.where(mask_bin)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    gray_crop = gray[y0:y1, x0:x1]
    mask_crop = mask_bin[y0:y1, x0:x1]
    gray_masked = gray_crop * mask_crop  # outside-mask pixels become 0

    # Reduce gray levels for a smaller GLCM
    levels = 32
    gray_q = (gray_masked / 256.0 * levels).astype(np.uint8)

    glcm = graycomatrix(gray_q, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=levels, symmetric=True, normed=True)

    return {
        'glcm_contrast': float(graycoprops(glcm, 'contrast').mean()),
        'glcm_homogeneity': float(graycoprops(glcm, 'homogeneity').mean()),
        'glcm_energy': float(graycoprops(glcm, 'energy').mean()),
        'glcm_correlation': float(graycoprops(glcm, 'correlation').mean()),
    }


def background_color_stats(image, mask):
    """
    Mean RGB of pixels OUTSIDE the bug mask.
    Captures the dominant background color (often a flower).
    """
    mask_bool = mask > 0
    bg_mask = ~mask_bool
    stats = {}
    for i, ch in enumerate(['r', 'g', 'b']):
        pixels = image[..., i][bg_mask]
        if pixels.size == 0:
            stats[f'bg_{ch}_mean'] = 0.0
        else:
            stats[f'bg_{ch}_mean'] = float(pixels.mean())
    return stats


# -----------------------------------------------------------------------------
# Master extraction function
# -----------------------------------------------------------------------------

def extract_all_features(image, mask):
    """
    Compute all features for a single (image, mask) pair.

    Parameters
    ----------
    image : np.ndarray of shape (H, W, 3), dtype uint8
        The RGB image.
    mask : np.ndarray of shape (H, W), dtype uint8 or bool
        The binary segmentation mask (>0 inside the bug).

    Returns
    -------
    dict mapping feature name -> float
    """
    feats = {}
    feats['pixel_ratio'] = bug_pixel_ratio(mask)
    feats.update(rgb_stats_in_mask(image, mask))
    feats.update(shape_symmetry(mask))
    feats.update(color_symmetry(image, mask))
    feats.update(shape_descriptors(mask))
    feats.update(hsv_stats_in_mask(image, mask))
    feats.update(texture_features(image, mask))
    feats.update(background_color_stats(image, mask))
    return feats
