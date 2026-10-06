'''
Selective Search implementation for the Computer Vision Project.

The code follows the algorithm of Uijlings et al.:
1. get an initial over-segmentation with Felzenszwalb,
2. describe every region by colour, texture, size and fill,
3. iteratively merge the most similar neighbouring regions,
4. return all generated bounding-box proposals.
'''
# -*- coding: utf-8 -*-
from __future__ import division

from itertools import combinations

import numpy as np
import skimage.color
import skimage.feature
import skimage.segmentation
import skimage.util


def _as_float_rgb(image):
    """Return an RGB float image in [0, 1]."""
    if image.ndim != 3:
        raise ValueError("Expected an image with three channels.")
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return skimage.util.img_as_float(image)


def generate_segments(im_orig, scale, sigma, min_size):
    """
    Task 5.1: Segment smallest regions using Felzenszwalb.

    The returned image has four channels. The first three channels are the RGB
    image, and the fourth channel stores the segment label for each pixel.
    """
    image = _as_float_rgb(im_orig)
    labels = skimage.segmentation.felzenszwalb(
        image,
        scale=scale,
        sigma=sigma,
        min_size=min_size,
    )
    return np.dstack((image, labels.astype(np.float64)))


def _hist_intersection(h1, h2):
    return float(np.minimum(h1, h2).sum())


def sim_colour(r1, r2):
    """Task 5.2: Colour similarity using histogram intersection."""
    return _hist_intersection(r1["hist_c"], r2["hist_c"])


def sim_texture(r1, r2):
    """Task 5.2: Texture similarity using histogram intersection."""
    return _hist_intersection(r1["hist_t"], r2["hist_t"])


def sim_size(r1, r2, imsize):
    """Task 5.2: Prefer merging smaller regions first."""
    return 1.0 - (r1["size"] + r2["size"]) / float(imsize)


def sim_fill(r1, r2, imsize):
    """Task 5.2: Prefer regions that tightly fill their joint bounding box."""
    min_x = min(r1["min_x"], r2["min_x"])
    min_y = min(r1["min_y"], r2["min_y"])
    max_x = max(r1["max_x"], r2["max_x"])
    max_y = max(r1["max_y"], r2["max_y"])
    bbsize = (max_x - min_x + 1) * (max_y - min_y + 1)
    return 1.0 - (bbsize - r1["size"] - r2["size"]) / float(imsize)


def calc_sim(r1, r2, imsize):
    return (
        sim_colour(r1, r2)
        + sim_texture(r1, r2)
        + sim_size(r1, r2, imsize)
        + sim_fill(r1, r2, imsize)
    )


def calc_colour_hist(img):
    """
    Task 5.2.5.1: Calculate a HSV colour histogram for one region.

    Input can be either HxWx3 or Nx3. We use 25 bins per channel, as in the
    selective search paper. The final vector is L1-normalized.
    """
    BINS = 25
    pixels = np.asarray(img)
    if pixels.size == 0:
        return np.zeros(BINS * 3, dtype=np.float64)
    pixels = pixels.reshape(-1, 3)

    hist_parts = []
    for channel in range(3):
        h, _ = np.histogram(
            pixels[:, channel],
            bins=BINS,
            range=(0.0, 1.0),
        )
        hist_parts.append(h.astype(np.float64))

    hist = np.concatenate(hist_parts)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def calc_texture_gradient(img):
    """
    Task 5.2.5.2: Calculate an LBP texture image.

    Uijlings et al. use Gaussian derivative filters. For this exercise skeleton
    we use local binary patterns (LBP), one channel at a time.
    """
    image = np.asarray(img)
    if image.shape[2] > 3:
        image = image[:, :, :3]

    ret = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float64)
    for channel in range(3):
        ret[:, :, channel] = skimage.feature.local_binary_pattern(
            skimage.util.img_as_ubyte(image[:, :, channel]),
            P=8,
            R=1,
            method="uniform",
        )
    return ret


def calc_texture_hist(img):
    """
    Task 5.2.5.3: Calculate a texture histogram for one region.

    LBP with P=8 and method='uniform' produces values in [0, 9]. We use ten
    bins per channel and L1-normalize the final vector.
    """
    BINS = 10
    pixels = np.asarray(img)
    if pixels.size == 0:
        return np.zeros(BINS * 3, dtype=np.float64)
    pixels = pixels.reshape(-1, 3)

    hist_parts = []
    for channel in range(3):
        h, _ = np.histogram(
            pixels[:, channel],
            bins=BINS,
            range=(0.0, 10.0),
        )
        hist_parts.append(h.astype(np.float64))

    hist = np.concatenate(hist_parts)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def extract_regions(img):
    '''
    Task 5.2.5: Generate the region data structure R.
    '''
    R = {}
    labels = img[:, :, 3].astype(np.int64)
    rgb = img[:, :, :3]
    hsv = skimage.color.rgb2hsv(rgb)
    texture_gradient = calc_texture_gradient(rgb)

    for label in np.unique(labels):
        mask = labels == label
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue

        region_pixels_hsv = hsv[mask]
        region_pixels_texture = texture_gradient[mask]

        R[float(label)] = {
            "min_x": int(xs.min()),
            "min_y": int(ys.min()),
            "max_x": int(xs.max()),
            "max_y": int(ys.max()),
            "size": int(mask.sum()),
            "hist_c": calc_colour_hist(region_pixels_hsv),
            "hist_t": calc_texture_hist(region_pixels_texture),
            "labels": [float(label)],
        }

    return R


def _bbox_intersects(a, b):
    """True if two bounding boxes overlap or touch."""
    return not (
        a["max_x"] < b["min_x"]
        or b["max_x"] < a["min_x"]
        or a["max_y"] < b["min_y"]
        or b["max_y"] < a["min_y"]
    )


def extract_neighbours(regions):
    """Task 5.3: Find neighbouring/intersecting region pairs."""
    neighbours = []
    for (label_a, region_a), (label_b, region_b) in combinations(regions.items(), 2):
        if _bbox_intersects(region_a, region_b):
            neighbours.append(((label_a, region_a), (label_b, region_b)))
    return neighbours


def merge_regions(r1, r2):
    """Task 5.4: Merge two regions and combine their descriptors."""
    new_size = r1["size"] + r2["size"]
    rt = {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "size": new_size,
        "hist_c": (
            r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]
        ) / float(new_size),
        "hist_t": (
            r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]
        ) / float(new_size),
        "labels": r1["labels"] + r2["labels"],
    }
    return rt


def selective_search(image_orig, scale=1.0, sigma=0.8, min_size=50):
    '''
    Selective Search for Object Recognition by Uijlings et al.
    '''
    assert image_orig.shape[2] == 3, "Please use image with three channels."
    imsize = image_orig.shape[0] * image_orig.shape[1]

    # Task 5.1: Initial segmentation.
    image = generate_segments(image_orig, scale, sigma, min_size)
    if image is None:
        return None, {}

    # Task 5.2: Region extraction and descriptors.
    R = extract_regions(image)

    # Task 5.3: Neighbour extraction.
    neighbours = extract_neighbours(R)

    # Initial similarities.
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = calc_sim(ar, br, imsize)

    # Hierarchical grouping.
    while S:
        # Most similar pair.
        i, j = max(S.items(), key=lambda item: item[1])[0]

        # Task 5.4: Merge regions.
        t = max(R.keys()) + 1.0
        R[t] = merge_regions(R[i], R[j])

        # Task 5.5: Mark old similarities related to i or j.
        keys_to_delete = [key for key in S if i in key or j in key]

        # Candidate neighbours of the new region are old neighbours of i or j.
        candidate_neighbours = set()
        for a, b in keys_to_delete:
            if a not in (i, j):
                candidate_neighbours.add(a)
            if b not in (i, j):
                candidate_neighbours.add(b)

        # Task 5.6: Remove old similarities.
        for key in keys_to_delete:
            del S[key]

        # Task 5.7: Calculate new similarities with the merged region.
        for n in candidate_neighbours:
            if n in R and _bbox_intersects(R[t], R[n]):
                S[(t, n)] = calc_sim(R[t], R[n], imsize)

    # Task 5.8: Generate final bounding box proposals.
    regions = []
    for region in R.values():
        x = region["min_x"]
        y = region["min_y"]
        w = region["max_x"] - region["min_x"] + 1
        h = region["max_y"] - region["min_y"] + 1
        regions.append({
            "rect": (x, y, w, h),
            "size": region["size"],
            "labels": region["labels"],
        })

    return image, regions
