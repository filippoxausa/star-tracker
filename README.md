# Star Tracker

An educational image processing project that analyzes a sequence of images from a "malfunctioning" star tracker. The goal is to process a stream of noisy images to detect stars and estimate the satellite's motion.

---

## Project Description

The input consists of a sequence of **50 grayscale frames** (numbered `0.jpg` through `49.jpg`) containing stars mixed with significant noise.
This project implements a complete pipeline that automatically:
1. **Loads and Visualizes** the raw image sequence.
2. **Cleans** the noise using morphological filters.
3. **Detects** stars, calculating their bounding boxes and centroids.
4. **Tracks** the satellite's motion by calculating the average displacement of stars between consecutive frames.

<img width="400" height="200" alt="Denoising Comparison_screenshot_14 07 2026" src="https://github.com/user-attachments/assets/3f61b548-9d59-438a-ad58-d32a59b32e97" />

<img width="200" height="200" alt="Star Detection_screenshot_14 07 2026" src="https://github.com/user-attachments/assets/4c337485-b13d-47d9-b7a7-d5f7dc2428d3" />

---

## Requirements & Installation

**Prerequisites:**
- Python 3.8+
- A dataset folder `images/` containing the 50 provided `.jpg` frames.

**Installation:**

You can install the required libraries (`opencv-python` and `numpy`) using the provided `requirements.txt` file:

`pip install -r requirements.txt`

Alternatively, you can install them manually:

`pip install opencv-python numpy`


Clone this repository, navigate to the project root, and ensure the `images/` folder is populated with the star tracker frames.

---

## Project Structure

Recommended structure:

```
├── images/ # Source frames: 0.jpg to 49.jpg
├── main.py # Main script containing the full pipeline
└── requirements.txt # List of dependencies
```

Upon execution, the script will automatically generate two output files in the root directory:
- `bounding_boxes.txt`: Contains the coordinates of detected stars for each frame.
- `motion_log.txt`: Contains the calculated displacement vector of the satellite between consecutive frames.

---

## How It Works

The `main.py` script runs a continuous pipeline. It sequentially processes the images using the following steps:

### 1. Image Loading & Sorting
The script reads all `.jpg` files from the `images/` directory and sorts them numerically to ensure the sequence `0, 1, 2...` is respected, preserving the temporal order of the satellite's movement.

### 2. Denoising (Pre-processing)
Each image undergoes binary thresholding to separate bright pixels from the background. Morphological operations (`MORPH_OPEN`) are then applied to remove small and isolated noise pixels (erosion + dilation).

### 3. Star Detection & Logging
The script identifies contours in the cleaned binary images. For every detected star:
- A **Bounding Box** `(x1, y1, x2, y2)` is calculated.
- The **Centroid** (center of mass) is computed using image moments.
All bounding box coordinates for a frame are written to `result/bounding_boxes.txt` on a single line (format: `x1 y1 x2 y2 ...`).

### 4. Motion Tracking
Finally, the script compares the positions of star centroids between Frame `t` and Frame `t+1`. By matching stars based on proximity, it calculates the displacement vectors `(Δx, Δy)` of the star field. A median filter is applied to these vectors to robustly estimate the satellite's frame-to-frame motion, ignoring outliers caused by noise. The results (horizontal and vertical shift) are logged to `result/motion_log.txt`.

---

## Usage

Simply run the main script:

`python main.py`

The program will display a visualization window showing the processing steps (denoising, star detection with bounding boxes, and motion vectors). Press `q` to close the visualization (and elaboration) window early.
