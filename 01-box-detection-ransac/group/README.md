# Exercise 1 Solution – Box Detection

## 1. What is this code about?

This project implements a solution for **Exercise 1.1 / 1.2: Box Detection** from the FAU Computer Vision Project.

The goal of this work is to process Kinect sensor data and automatically estimate the physical dimensions of a box placed on the floor. The input data are provided as MATLAB `.mat` files. Each file contains an amplitude image, a distance image, and a 3D point cloud.

The implemented pipeline detects the floor plane, removes the floor points, detects the top plane of the box, extracts the largest connected component as the final box-top region, and estimates the box **length**, **width**, and **height**.

A key requirement of the exercise is to implement RANSAC manually instead of using a ready-made RANSAC implementation. This solution follows that requirement by using a self-written RANSAC algorithm for plane detection.

---

## 2. Folder structure

The project folder is organized as follows:

```text
ProjectCV_EX_1/
│
├── box_detection_solution.py
├── discussion.md
├── README.md
│
├── data/
│   ├── example1kinect.mat
│   ├── example2kinect.mat
│   ├── example3kinect.mat
│   ├── example4kinect.mat
│   └── Test/
│
└── results/
    ├── summary.json
    ├── example1_inputs.png
    ├── example1_masks.png
    ├── example1_3d.png
    ├── example1_presentation.png
    ├── example2_inputs.png
    ├── example2_masks.png
    ├── example2_3d.png
    ├── example2_presentation.png
    ├── example3_inputs.png
    ├── example3_masks.png
    ├── example3_3d.png
    ├── example3_presentation.png
    ├── example4_inputs.png
    ├── example4_masks.png
    ├── example4_3d.png
    └── example4_presentation.png
```

### Main files and folders

| File / Folder | Description |
|---|---|
| `box_detection_solution.py` | Main Python script containing the complete implementation. |
| `discussion.md` | Discussion file for Exercise 1.2. |
| `data/` | Folder containing the Kinect `.mat` input files. |
| `results/` | Folder containing generated images and the final JSON summary. |
| `results/summary.json` | Stores the numerical results for all four examples. |
| `example*_inputs.png` | Input visualization: amplitude image, distance image, and 3D point cloud. |
| `example*_masks.png` | Mask visualization: floor mask, filtered floor mask, top mask, and final box-top region. |
| `example*_3d.png` | 3D visualization of the scene, floor, box top, and estimated corners. |
| `example*_presentation.png` | Clean presentation-style visualization similar to the exercise sheet. |

---

## 3. How to reproduce the work

### 3.1 Create and activate a virtual environment

From the project folder, create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

### 3.2 Install required packages

Install the required Python packages:

```bash
pip install numpy scipy matplotlib
```

The solution uses:

- `numpy` for numerical computation and array operations
- `scipy` for loading `.mat` files and image morphology
- `matplotlib` for generating and saving visualizations

The solution does **not** use `scikit-learn` RANSAC. The RANSAC algorithm is implemented manually in the code.

### 3.3 Prepare the input data

Place the provided Kinect `.mat` files inside the `data/` folder:

```text
data/example1kinect.mat
data/example2kinect.mat
data/example3kinect.mat
data/example4kinect.mat
```

The script uses the following input and output folders:

```python
input_dir = "data"
out_dir = "results"
```

Therefore, the code reads the input files from `data/` and writes all generated results into `results/`.

### 3.4 Run the script

Run the code with:

```bash
python3 box_detection_solution.py
```

After running, the script creates or updates the `results/` folder and saves:

- input visualizations
- mask visualizations
- 3D visualizations
- presentation-style visualizations
- `summary.json`

---

## 4. Method overview

The method is directly based on the exercise description:

1. Load one of the example files and visualize amplitude image, distance image, and 3D point cloud.  
2. Ignore invalid points whose z-value is `0`.  
3. Run self-written RANSAC on the 3D point cloud to detect the **floor plane**.  
4. Convert floor inliers into a mask image.  
5. Improve the mask using morphological filtering. The exercise explicitly asks to evaluate which operators work best. In this solution, **closing** is used first to fill small holes in the floor mask, then **opening** is used to remove small noise blobs.  
6. Remove the floor points and run RANSAC again on the remaining 3D points to get the **top plane of the box**.  
7. Use connected components and keep the **largest connected component**, because the sheet says it can be assumed to delimit the box top.  
8. Estimate the **height** using the distance between top-plane points and the floor plane, with plane offset as a cross-check.  
9. Estimate **length** and **width** from a PCA-aligned oriented bounding box on the top plane.  
10. Visualize masks, planes, and corners.  

## 5. Work overview

The complete workflow of the code is:

```text
Load Kinect .mat file
        ↓
Visualize input data
        ↓
Run RANSAC to detect floor plane
        ↓
Convert floor inliers into image mask
        ↓
Clean floor mask using morphology
        ↓
Remove floor points
        ↓
Run RANSAC again to detect box top plane
        ↓
Convert top-plane inliers into image mask
        ↓
Clean top mask
        ↓
Keep largest connected component
        ↓
Estimate box height, length, and width
        ↓
Save figures and JSON summary
```

### 5.1 Loading the Kinect data

Each `.mat` file contains three main data components:

- amplitude image
- distance image
- 3D point cloud

The function `load_example()` loads these values for each example.

### 5.2 Input visualization

The function `save_input_visualization()` saves an overview figure containing:

- the amplitude image
- the distance image
- a subsampled 3D point cloud

This step is useful for inspecting the raw input data before applying plane detection.

### 5.3 Plane detection using self-written RANSAC

The function `ransac_plane()` is used to detect dominant planes in the 3D point cloud.

The algorithm randomly selects three points, forms a possible plane, and counts how many other points lie close to that plane. The plane with the highest number of matching points is selected as the best plane.

This is first used to detect the floor plane.

### 5.4 Floor mask generation and cleaning

After detecting the floor plane, the floor inlier points are converted back into a 2D image mask.

The mask is cleaned using morphological operations:

- **closing** fills small holes in the mask
- **opening** removes small isolated noise regions

This produces a cleaner floor region.

### 5.5 Box top detection

After the floor is removed, RANSAC is applied again to the remaining points.

This second RANSAC step detects the top plane of the box.

The resulting top-plane mask is also cleaned using morphological operations.

### 5.6 Largest connected component

The raw top-plane mask may contain small noisy planar regions.

Therefore, the function `largest_connected_component()` keeps only the largest connected region.

This largest component is treated as the actual box top.

### 5.7 Dimension estimation

The box dimensions are estimated from the final detected box-top points.

The height is calculated using the average distance between the box-top points and the floor plane.

A second height value is also computed using the difference between the floor-plane and top-plane offsets as a consistency check.

The length and width are estimated from an oriented bounding box fitted to the detected top-plane points.

### 5.8 Output visualizations

The script saves four types of output images for each example.

#### Input visualization

```text
example*_inputs.png
```

This image shows:

- amplitude image
- distance image
- subsampled 3D point cloud

#### Mask visualization

```text
example*_masks.png
```

This image shows:

- distance image
- raw floor mask
- filtered floor mask
- top-plane mask
- final detected box-top region

#### 3D visualization

```text
example*_3d.png
```

This image shows:

- full scene point cloud
- detected floor points
- detected box-top points
- estimated top corners

#### Presentation visualization

```text
example*_presentation.png
```

This image shows a clean presentation-style result with:

- blue background
- green floor
- red box top
- cyan box outline
- top, left, right, and bottom labels

---

## 6. Results

The script was executed using:

```bash
python3 box_detection_solution.py
```

The terminal output was:

```text
Example 1: L=0.649 m, W=0.470 m, H=0.189 m (plane distance cross-check: 0.193 m)
Example 2: L=0.472 m, W=0.402 m, H=0.189 m (plane distance cross-check: 0.192 m)
Example 3: L=0.474 m, W=0.399 m, H=0.192 m (plane distance cross-check: 0.190 m)
Example 4: L=0.470 m, W=0.374 m, H=0.184 m (plane distance cross-check: 0.185 m)
```

### 6.1 Measured box dimensions

| Example | Length (m) | Width (m) | Height from point-to-floor distance (m) | Height from plane offset cross-check (m) |
|---:|---:|---:|---:|---:|
| 1 | 0.649 | 0.470 | 0.189 | 0.193 |
| 2 | 0.472 | 0.402 | 0.189 | 0.192 |
| 3 | 0.474 | 0.399 | 0.192 | 0.190 |
| 4 | 0.470 | 0.374 | 0.184 | 0.185 |

The two height estimates are close to each other for all four examples. This indicates that the detected floor and top planes are reasonably consistent.

### 6.2 Detailed result summary

#### Example 1

```text
Length: 0.6488 m
Width:  0.4705 m
Height: 0.1890 m
Plane-distance height cross-check: 0.1934 m
Floor inliers: 140477
Valid points: 202007
Top component pixels: 37111
```

#### Example 2

```text
Length: 0.4723 m
Width:  0.4019 m
Height: 0.1892 m
Plane-distance height cross-check: 0.1923 m
Floor inliers: 161476
Valid points: 199462
Top component pixels: 22941
```

#### Example 3

```text
Length: 0.4739 m
Width:  0.3991 m
Height: 0.1921 m
Plane-distance height cross-check: 0.1902 m
Floor inliers: 167987
Valid points: 198597
Top component pixels: 15857
```

#### Example 4

```text
Length: 0.4703 m
Width:  0.3742 m
Height: 0.1843 m
Plane-distance height cross-check: 0.1849 m
Floor inliers: 165628
Valid points: 202430
Top component pixels: 28376
```

---

## 7. Discussion

The implemented method successfully detects the floor and the box top from the provided Kinect point clouds.

RANSAC is suitable for this task because the point cloud contains noise, invalid measurements, and points from multiple surfaces. By searching for the dominant plane, the method can robustly identify the floor even when the box and other scene points are present.

Morphological filtering improves the quality of the detected masks. Closing fills small gaps in the detected regions, while opening removes small isolated noise components. This makes the final floor and top-plane masks more stable.

The largest connected component step is important for the box-top detection. Without this step, small noisy planar regions could remain in the result. By keeping only the largest connected region, the method obtains a cleaner estimate of the actual box top.

The height estimation is consistent because the point-to-floor distance and the plane-offset cross-check produce similar values for all examples. The length and width values are also stable for examples 2, 3, and 4. Example 1 has a larger estimated length and width, which suggests that either the visible top region is larger in that scene or the detected component covers a larger area.

Overall, the solution follows the exercise requirements and provides a complete pipeline for loading Kinect data, detecting planes, cleaning masks, estimating box dimensions, and visualizing the results.