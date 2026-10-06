from __future__ import division

import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy as np
import heapq

def generate_segments(im_orig, scale, sigma, min_size):

    image_float = skimage.util.img_as_float(im_orig)
    labels = skimage.segmentation.felzenszwalb(
        image_float, scale=scale, sigma=sigma, min_size=min_size
    )

    image = np.zeros((im_orig.shape[0], im_orig.shape[1], 4), dtype=np.float32)
    image[:, :, :3] = image_float
    image[:, :, 3] = labels

    return image

def sim_colour(r1, r2):
    return np.minimum(r1["hist_c"], r2["hist_c"]).sum()


def sim_texture(r1, r2):
   
    return np.minimum(r1["hist_t"], r2["hist_t"]).sum()


def sim_size(r1, r2, imsize):

    return 1.0 - (r1["size"] + r2["size"]) / imsize


def sim_fill(r1, r2, imsize):
    
    min_x = min(r1["min_x"], r2["min_x"])
    min_y = min(r1["min_y"], r2["min_y"])
    max_x = max(r1["max_x"], r2["max_x"])
    max_y = max(r1["max_y"], r2["max_y"])

    bbox_size = (max_x - min_x + 1) * (max_y - min_y + 1)
    return 1.0 - (bbox_size - r1["size"] - r2["size"]) / imsize

def calc_sim(r1, r2, imsize):
    return (sim_colour(r1, r2) + sim_texture(r1, r2)
            + sim_size(r1, r2, imsize) + sim_fill(r1, r2, imsize))

def calc_colour_hist(img):

    BINS = 25
    hist = np.zeros(BINS * 3, dtype=np.float32)
    if img.size == 0:
        return hist

    for channel in range(3):
        channel_hist, _ = np.histogram(
            img[:, channel], bins=BINS, range=(0.0, 1.0)
        )
        start = channel * BINS
        hist[start:start + BINS] = channel_hist.astype(np.float32)

    hist_sum = hist.sum()
    if hist_sum > 0:
        hist = hist / hist_sum

    return hist

def calc_texture_gradient(img):
 
    img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)
    for channel in range(img.shape[2]):
        ret[:, :, channel] = skimage.feature.local_binary_pattern(
            img_uint8[:, :, channel], P=8, R=1, method="uniform"
        )

    return ret

def calc_texture_hist(img):
    
    BINS = 10
    hist = np.zeros(BINS * 3, dtype=np.float32)
    if img.size == 0:
        return hist

    for channel in range(3):
        channel_hist, _ = np.histogram(
            img[:, channel], bins=BINS, range=(0.0, BINS)
        )
        start = channel * BINS
        hist[start:start + BINS] = channel_hist.astype(np.float32)

    hist_sum = hist.sum()
    if hist_sum > 0:
        hist = hist / hist_sum

    return hist

def extract_regions(img):
 
    R = {}
    hsv = skimage.color.rgb2hsv(img[:, :, :3])
    texture_gradient = calc_texture_gradient(hsv)
    labels = img[:, :, 3].astype(np.int32)
    height, width = labels.shape
    flat_labels = labels.ravel()
    hsv_flat = hsv.reshape(-1, 3)
    texture_flat = texture_gradient.reshape(-1, 3)

    order = np.argsort(flat_labels, kind="mergesort")
    sorted_labels = flat_labels[order]
    split_points = np.flatnonzero(np.diff(sorted_labels)) + 1

    for indices in np.split(order, split_points):
        if indices.size == 0:
            continue
        label = int(flat_labels[indices[0]])
        ys, xs = np.divmod(indices, width)

        R[label] = {
            "min_x": int(xs.min()),
            "min_y": int(ys.min()),
            "max_x": int(xs.max()),
            "max_y": int(ys.max()),
            "labels": [int(label)],
            "size": int(indices.size),
            "hist_c": calc_colour_hist(hsv_flat[indices]),
            "hist_t": calc_texture_hist(texture_flat[indices]),
        }

    return R

def extract_neighbours(regions):

    def intersect(a, b):
        return not (
            a["max_x"] + 1 < b["min_x"]
            or b["max_x"] + 1 < a["min_x"]
            or a["max_y"] + 1 < b["min_y"]
            or b["max_y"] + 1 < a["min_y"]
        )

    # Hint 1: List of neighbouring regions
    # Hint 2: The function intersect has been written for you and is required to check neighbours
    neighbours = []
    if not regions:
        return neighbours

    widths = [r["max_x"] - r["min_x"] + 1 for r in regions.values()]
    heights = [r["max_y"] - r["min_y"] + 1 for r in regions.values()]
    cell_size = max(16, int(np.median(widths + heights)))
    grid = {}

    for region_id, region in regions.items():
        min_cell_x = region["min_x"] // cell_size
        max_cell_x = region["max_x"] // cell_size
        min_cell_y = region["min_y"] // cell_size
        max_cell_y = region["max_y"] // cell_size
        for cell_y in range(min_cell_y, max_cell_y + 1):
            for cell_x in range(min_cell_x, max_cell_x + 1):
                grid.setdefault((cell_x, cell_y), []).append(region_id)

    checked = set()
    for cell_region_ids in grid.values():
        for index, region_a_id in enumerate(cell_region_ids):
            for region_b_id in cell_region_ids[index + 1:]:
                key = (region_a_id, region_b_id)
                if key in checked:
                    continue
                checked.add(key)
                a = regions[region_a_id]
                b = regions[region_b_id]
                if intersect(a, b):
                    neighbours.append(((region_a_id, a), (region_b_id, b)))


    return neighbours


def _make_similarity_key(region_a_id, region_b_id):
    return (
        region_a_id,
        region_b_id
    ) if region_a_id < region_b_id else (
        region_b_id,
        region_a_id
    )


def _push_similarity(S, heap, region_a_id, region_b_id, region_a, region_b, imsize):
    key = _make_similarity_key(region_a_id, region_b_id)
    similarity = calc_sim(region_a, region_b, imsize)
    S[key] = similarity
    heapq.heappush(heap, (-similarity, key))

def merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"]
    rt = {}
    rt["min_x"] = min(r1["min_x"], r2["min_x"])
    rt["min_y"] = min(r1["min_y"], r2["min_y"])
    rt["max_x"] = max(r1["max_x"], r2["max_x"])
    rt["max_y"] = max(r1["max_y"], r2["max_y"])
    rt["size"] = new_size
    rt["hist_c"] = (
        r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]
    ) / new_size
    rt["hist_t"] = (
        r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]
    ) / new_size
    rt["labels"] = r1["labels"] + r2["labels"]

    return rt


def selective_search(image_orig, scale=1.0, sigma=0.8, min_size=50):
    
    assert image_orig.shape[2] == 3, "Please use image with three channels."
    imsize = image_orig.shape[0] * image_orig.shape[1]

    image = generate_segments(image_orig, scale, sigma, min_size)

    if image is None:
        return None, {}

    R = extract_regions(image)

    
    neighbours = extract_neighbours(R)

    S = {}
    heap = []
    for (ai, ar), (bi, br) in neighbours:
        _push_similarity(S, heap, ai, bi, ar, br, imsize)

    next_region_id = max(R.keys()) + 1 if R else 0

    while S != {}:

        while heap:
            negative_similarity, key = heapq.heappop(heap)
            if S.get(key) == -negative_similarity:
                i, j = key
                break
        else:
            break

        t = next_region_id
        next_region_id += 1
        R[t] = merge_regions(R[i], R[j])

        key_to_delete = []
        regions_to_compare = set()
        for key in S.keys():
            if i in key or j in key:
                key_to_delete.append(key)
                other_region = key[1] if key[0] in (i, j) else key[0]
                if other_region not in (i, j):
                    regions_to_compare.add(other_region)


        for key in key_to_delete:
            del S[key]


        for region_id in regions_to_compare:
            _push_similarity(S, heap, t, region_id, R[t], R[region_id], imsize)


    regions = []
    seen_rects = set()
    for region in R.values():
        rect = (
            int(region["min_x"]),
            int(region["min_y"]),
            int(region["max_x"] - region["min_x"] + 1),
            int(region["max_y"] - region["min_y"] + 1),
        )
        if rect in seen_rects:
            continue
        seen_rects.add(rect)
        regions.append({
            "rect": rect,
            "size": int(region["size"]),
            "labels": region["labels"],
        })


    return image, regions
