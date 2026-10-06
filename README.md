# Computer Vision Exercises

This repo collects my work from the Computer Vision Project course at FAU Erlangen-Nürnberg (summer term 2026).
Every exercise had a group part, which I did with teammates, and an individual part where I extended the topic on my own.
Everything is written in Python. The datasets belong to the course, so they are not included here, but each exercise README explains where the data goes and how to run the code.

| # | Exercise | What I worked with | Main result |
|---|---|---|---|
| 01 | [Box detection with RANSAC](01-box-detection-ransac/) | Kinect point clouds, my own RANSAC, MLESAC and preemptive RANSAC, morphology | Box height of about 0.19 m found consistently in all 4 scenes |
| 02 | [Demosaicing and HDR](02-demosaicing-hdr/) | Bayer demosaicing, tone curves, gray world white balance, HDR merging, iCAM06, camera response from JPGs | A complete RAW to display pipeline and my own camera response curve |
| 03 | [Writer identification](03-writer-identification/) | SIFT, VLAD, normalisation, PCA whitening, Exemplar SVM, DINOv2 | Top 1 accuracy of 89.6 % and mAP of 0.770 |
| 04 | [Selective Search and object detection](04-selective-search-detection/) | Felzenszwalb segmentation, hierarchical grouping, HOG and colour features, linear SVM, NMS | Region proposals on art history images and a balloon detector with AP@0.5 of 0.30 |
| 05 | [Face recognition and open set recognition](05-face-recognition/) | MTCNN, tracking, face embeddings, k nearest neighbours, k means, DIR curves, pseudo labels | 96.3 % accuracy on known people while rejecting all unknown people |

<p align="center">
  <img src="02-demosaicing-hdr/group/results/final_result_process_raw.jpg" width="32%">
  <img src="01-box-detection-ransac/group/results/example1_presentation.png" width="32%">
  <img src="04-selective-search-detection/group/results/arthist/annunciation1.jpg" width="32%">
</p>

## How the folders are set up

```
NN-exercise-name/
├── README.md        what the exercise is about, results and how to run it
├── group/           the team solution with code, results and report
└── individual/      my own extension with code, results and report
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r <exercise>/<group or individual>/requirements.txt
```

## Credits

The group parts were team work, and my teammates are named in each exercise README.
The exercise sheets, lecture slides and datasets come from the course organisers and are not part of this repo.

## Contact

Simu Sultana, [GitHub](https://github.com/Simu-Sultana), [LinkedIn](https://linkedin.com/in/shimu-sultana)
