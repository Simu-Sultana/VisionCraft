# Individual Exercise 1 – Box Detection with RANSAC Variants

> Note: in this repo the results are stored in `results/`. The script writes to `results_individual/` by default, so pass `--out-dir results` if you want them in the same place.

## 1. Project description

This project is the individual extension of **Exercise 1: Box Detection** from the Computer Vision Project.

The task is to estimate the size of a box from Kinect distance data. Each input example contains:

- an amplitude image,
- a distance image,
- and a registered 3D point cloud.

The group solution already implemented the basic box-detection pipeline using RANSAC. This individual version extends that pipeline by implementing and evaluating three self-written plane-fitting methods:

1. **Standard RANSAC**
2. **MLESAC-style RANSAC**
3. **Preemptive RANSAC**

The script also performs hyperparameter tuning, runtime analysis, and a simple runtime hypothesis test.

The implementation does **not** use scikit-learn's RANSAC. The RANSAC variants are implemented manually.

---

## 2. Expected folder structure

The project folder should look like this:

```text
Exercise_1/
│
├── box_detection_individual.py
├── README_individual.md
├── individual_discussion.md
│
├── data/
│   ├── example1kinect.mat
│   ├── example2kinect.mat
│   ├── example3kinect.mat
│   └── example4kinect.mat
│
└── results_individual/
    └── generated automatically after running the script
```

The `data/` folder must contain the provided Kinect `.mat` files. The code reads the input files from this folder by default.

---

## 3. Requirements

The code was written for Python 3 and uses the following packages:

```text
numpy
scipy
matplotlib
```

Install the required packages with:

```bash
pip install numpy scipy matplotlib
```

Optional but recommended:

```bash
pip install scipy
```

`scipy` is already required for loading `.mat` files and morphology. If available, `scipy.stats` is also used for the paired runtime hypothesis test.

---

## 4. How to run the code

From the project folder, run:

```bash
python box_detection_individual.py --mode full
```

On Windows PowerShell, use:

```powershell
python .\box_detection_individual.py --mode full
```

This runs:

1. method comparison,
2. hyperparameter tuning,
3. runtime analysis,
4. hypothesis test.

All results are saved in:

```text
results_individual/
```

---

## 5. Faster commands for testing

The full run can take several minutes because it runs many combinations of examples, methods, thresholds, and iteration counts.

For a quick check, run only the method comparison:

```bash
python box_detection_individual.py --mode compare
```

On Windows:

```powershell
python .\box_detection_individual.py --mode compare
```

This creates the main detection outputs for:

```text
ransac
mlesac
preemptive
```

For only hyperparameter tuning:

```bash
python box_detection_individual.py --mode tune
```

For only runtime analysis:

```bash
python box_detection_individual.py --mode timing
```

---

## 6. Command-line options

The script supports several command-line options.

### Input and output folders

```bash
python box_detection_individual.py --input-dir data --out-dir results_individual
```

Defaults:

```text
--input-dir data
--out-dir results_individual
```

### Select examples

```bash
python box_detection_individual.py --examples 1,2,3,4
```

Example: run only example 1:

```bash
python box_detection_individual.py --mode compare --examples 1
```

### Select methods

```bash
python box_detection_individual.py --methods ransac,mlesac,preemptive
```

Example: run only standard RANSAC:

```bash
python box_detection_individual.py --mode compare --methods ransac
```

### Change thresholds and iterations

```bash
python box_detection_individual.py --floor-threshold 0.025 --top-threshold 0.025 --max-iterations 300
```

### Change tuning grid

```bash
python box_detection_individual.py --mode tune --threshold-grid 0.015,0.020,0.025,0.030,0.035 --iteration-grid 150,300,600
```

---

## 7. Method overview

The full detection pipeline is:

```text
Load Kinect .mat file
        ↓
Visualize input data
        ↓
Remove invalid 3D points where z = 0
        ↓
Fit the floor plane using selected RANSAC variant
        ↓
Convert floor inliers into a 2D mask
        ↓
Clean the mask using morphological closing and opening
        ↓
Remove floor points
        ↓
Fit the box-top plane using selected RANSAC variant
        ↓
Convert top-plane inliers into a 2D mask
        ↓
Clean the top mask
        ↓
Keep the largest connected component
        ↓
Estimate box height, length, and width
        ↓
Save visualizations and numerical results
```

---

## 8. RANSAC variants

### 8.1 Standard RANSAC

Standard RANSAC randomly samples three points, estimates a plane, and counts how many points lie within the inlier threshold. The plane with the largest number of inliers is selected.

This is the baseline method.

### 8.2 MLESAC-style RANSAC

MLESAC uses a likelihood-based score instead of only counting inliers. It considers how close points are to the plane, so a model with smaller residuals can be preferred even if the inlier count is similar.

This can improve robustness when the data contains noisy or borderline points.

### 8.3 Preemptive RANSAC

Preemptive RANSAC generates many candidates and evaluates them gradually. After each scoring block, weak candidates are removed. This can reduce runtime because poor candidates are not evaluated on all points.

The implementation avoids pruning candidates before they have been evaluated on comparable evidence.

---

## 9. Output files

After running the script, the output folder `results_individual/` contains several result files.

### 9.1 Numerical results

```text
individual_method_comparison.csv
individual_method_comparison.json
individual_hyperparameter_tuning.csv
individual_hyperparameter_tuning.json
individual_best_hyperparameters.csv
individual_timing_runs.csv
individual_timing_runs.json
individual_timing_summary.json
individual_hypothesis_test_runtime.json
```

### 9.2 Input visualizations

```text
example1_inputs.png
example2_inputs.png
example3_inputs.png
example4_inputs.png
```

Each image shows:

- amplitude image,
- distance image,
- subsampled 3D point cloud.

### 9.3 Detection visualizations

For each example and method:

```text
example<id>_<method>_masks.png
example<id>_<method>_3d.png
example<id>_<method>_presentation.png
```

Example:

```text
example1_ransac_masks.png
example1_ransac_3d.png
example1_ransac_presentation.png
example1_mlesac_presentation.png
example1_preemptive_presentation.png
```

### 9.4 Method comparison plots

For each example:

```text
example<id>_method_height_comparison.png
example<id>_method_runtime_comparison.png
example<id>_method_quality_comparison.png
```

These compare RANSAC, MLESAC, and preemptive RANSAC.

### 9.5 Hyperparameter tuning plots

For each example and method:

```text
example<id>_<method>_tuning_dimensions.png
example<id>_<method>_tuning_runtime_quality.png
```

These show how the threshold and iteration budget affect dimensions, runtime, and quality score.

### 9.6 Runtime plot

```text
individual_timing_summary.png
```

This compares the average runtime of the three methods.

---

## 10. Conclusion

This individual solution extends the original box-detection pipeline by comparing three RANSAC variants. Standard RANSAC provides the baseline, MLESAC improves model scoring by using residual likelihood, and preemptive RANSAC focuses on speed by pruning weak candidates early.

The additional hyperparameter tuning, runtime evaluation, and hypothesis test make the submission more complete because they show not only that the algorithm works, but also how its behaviour changes under different settings.
