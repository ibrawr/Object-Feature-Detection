# Object Feature Detection

> CSCI435 Computer Vision Algorithms and Systems · University of Wollongong in Dubai · Python · OpenCV · Haar Cascades · SIFT · SVM · ResNet50 · Google Colab

This repository contains an academic computer vision project that explores how classical image processing and machine learning techniques can be used to analyse real-world images. The system brings together multiple computer vision modules, including Dubai landmark detection, face and facial feature detection, frontal car detection, time-of-day classification, image similarity retrieval, image quality assessment, and manual image annotation.

The project was built around a practical question: how far can lightweight and interpretable computer vision methods go without relying fully on deep learning? Most modules were developed using OpenCV, Haar cascade classifiers, SIFT features, color space analysis, simple thresholding, and supervised classification. A ResNet50 feature extractor was also used in the image similarity module to compare classical features with deeper visual representations.

The final system is modular, easy to understand, and designed for experimentation. It is not a production-grade recognition system, but it demonstrates a complete computer vision workflow from dataset creation and model training to evaluation, interface design, and reflection on limitations.

* * *

## What This Project Does

  * Detects four major Dubai landmarks: Burj Khalifa, Burj Al Arab, Atlantis The Palm, and Dubai Frame
  * Trains custom Haar cascade classifiers using positive and negative landmark images
  * Tests alternative landmark recognition methods using SIFT, RootSIFT, Bag-of-Visual-Words, MiniBatchKMeans, and SVM classification
  * Attempts to use the Google Landmark Detection Dataset v2 for template-based landmark recognition
  * Detects frontal faces using OpenCV's pretrained Haar cascade classifier
  * Detects facial features such as eyes and smiles within detected face regions
  * Detects frontal cars in traffic images using a pretrained Haar cascade model
  * Classifies images as day or night using RGB, HSV, and LAB brightness features
  * Compares threshold-based classification with a linear SVM model for time-of-day prediction
  * Retrieves visually similar images using color histograms, Canny edge maps, ResNet50 deep features, and combined weighted similarity
  * Filters near-duplicate retrieval results using perceptual hashing
  * Assesses image quality by detecting blur, exposure issues, and noise
  * Allows users to manually crop a region of interest using coordinate-based annotation
  * Provides a command-line menu that connects the main computer vision modules into one interactive workflow

* * *

## Project Objective

The main objective of this project was to design and evaluate a modular computer vision system using OpenCV and related image processing techniques.

The project focused on the following questions:

  * Can classical computer vision methods detect Dubai landmarks from a small custom dataset?
  * How reliable are Haar cascade classifiers for faces, eyes, smiles, cars, and landmarks?
  * Can simple brightness features classify whether an image was taken during the day or at night?
  * Do deep visual features improve image similarity retrieval compared with color and edge features alone?
  * Can image quality issues be flagged using lightweight OpenCV-based metrics?
  * What are the practical limitations of building an interactive computer vision system inside Google Colab?

* * *

## My Contributions

This was a group project for CSCI435. My contribution can be presented as:

  * Computer vision experimentation - contributed to testing and evaluating multiple OpenCV-based approaches across detection, classification, and image retrieval tasks
  * Landmark detection workflow - helped structure the landmark detection pipeline around positive and negative image samples, Haar cascade training, and model evaluation
  * Evaluation and analysis - contributed to interpreting model performance using accuracy, precision, recall, F1-score, confusion matrices, and qualitative output examples
  * Interface integration - helped consolidate the separate modules into an interactive menu-based workflow so each feature could be tested from one place
  * Documentation and reporting - contributed to the final report structure, methodology explanations, result summaries, limitations, and discussion of what worked and what failed

* * *

## Dataset

The project used a custom image dataset created for four Dubai landmarks:

  * Atlantis The Palm
  * Burj Al Arab
  * Burj Khalifa
  * Dubai Frame

Each landmark folder was split into positive and negative samples. Positive samples contained the target landmark, while negative samples contained other images that should not trigger the detector.

| Landmark | Original Positives | Original Negatives | Augmented Positives | Augmented Negatives |
|---|---:|---:|---:|---:|
| Atlantis The Palm | 15 | 62 | 59 | 95 |
| Burj Al Arab | 16 | 61 | 64 | 93 |
| Burj Khalifa | 25 | 51 | 99 | 83 |
| Dubai Frame | 15 | 62 | 60 | 94 |

The positive landmark images were manually annotated using OpenCV annotation tools. These annotations were then converted into `.vec` sample files and used to train Haar cascade classifiers.

* * *

## Methodology

### 1. Landmark Detection

The landmark detection module focused on identifying whether an input image contained one of the four Dubai landmarks.

Three approaches were explored:

  * Haar cascade classifiers
  * SIFT with Bag-of-Visual-Words and SVM classification
  * Google Landmark Detection Dataset v2 with OpenCV-based feature matching

The final selected approach was the Haar cascade method because it produced bounding boxes and gave the most practical results for this project. Each landmark had its own trained XML classifier. The system resized the input image, converted it to grayscale, and used `detectMultiScale3()` to detect landmark regions with confidence scores.

The detection settings were:

| Parameter | Value |
|---|---:|
| Scale factor | 1.1 |
| Minimum neighbors | 3 |
| Minimum size | 100 x 100 |
| Confidence threshold | 1.00 |

A strict threshold was used to reduce false positives. This improved precision, but it also meant that some valid landmarks were missed when the image had unusual angles, lighting, or occlusion.

### 2. SIFT, BoVW, and SVM Experiment

A second landmark recognition pipeline was built using local feature descriptors.

The pipeline used:

  * SIFT descriptors
  * RootSIFT normalization
  * MiniBatchKMeans for visual vocabulary creation
  * Bag-of-Visual-Words histograms
  * StandardScaler for feature scaling
  * RBF-kernel SVMs for classification

The training results looked strong, but the model did not generalize well to unseen images. It performed well on training samples but misclassified new images when lighting, viewpoint, or background changed. This made it less suitable than the Haar cascade approach for the final integrated system.

| Landmark | Average Positive Score | Average Negative Score | Training Accuracy |
|---|---:|---:|---:|
| Atlantis The Palm | 0.9638 | 0.0266 | 0.9733 |
| Burj Al Arab | 0.9024 | 0.0339 | 0.9733 |
| Burj Khalifa | 0.8937 | 0.0483 | 0.9595 |
| Dubai Frame | 0.8928 | 0.0709 | 0.9200 |

### 3. Google Landmark Dataset Attempt

The project also explored the Google Landmark Detection Dataset v2. The aim was to filter UAE-related landmark classes and use OpenCV feature matching against downloaded landmark templates.

This approach was not successful because the image download step failed. The metadata contained many image IDs that returned HTTP 404 errors, which meant there were no reliable template images available for matching. The dataset was also more suitable for deep learning retrieval pipelines than traditional OpenCV descriptor matching.

| Metric | Result |
|---|---|
| UAE landmark classes filtered | Around 255 |
| Successful image downloads | 0 |
| Descriptor matching | Not completed |
| Precision and recall | Not applicable |

### 4. Face Detection

Face detection was implemented using OpenCV's pretrained `haarcascade_frontalface_default.xml` classifier. Images were converted to grayscale before detection, and bounding boxes were drawn around detected faces.

| Metric | Value |
|---|---:|
| Accuracy | 0.86 |
| Macro average F1 | 0.86 |
| Weighted average F1 | 0.87 |

The detector performed well on clear frontal faces and achieved perfect recall for the face class. The main weakness was false positives in images with complex backgrounds or face-like patterns.

### 5. Eye and Smile Detection

The facial feature pipeline used pretrained Haar cascade models for eyes and smiles. These classifiers were applied inside detected face regions rather than across the whole image, which improved speed and reduced irrelevant detections.

| Feature | Accuracy | Positive Class F1-Score | Main Observation |
|---|---:|---:|---|
| Eye detection | 0.95 | 0.94 | Strong performance with a few false positives |
| Smile detection | 0.54 | 0.64 | High recall but weak precision |

Eye detection performed strongly, while smile detection struggled because smiles are more sensitive to expression, lighting, shadows, and dataset imbalance.

### 6. Frontal Car Detection

Car detection was implemented using a pretrained frontal car Haar cascade. The model was tested on 35 images, including 13 car images and 22 non-car images.

| Metric | Value |
|---|---:|
| Accuracy | 0.74 |
| Car precision | 0.62 |
| Car recall | 0.77 |
| Car F1-score | 0.69 |

The detector worked best for clear frontal vehicles. It struggled with angled vehicles, partial occlusion, and non-frontal traffic images.

### 7. Time-of-Day Classification

This module classified images as day or night using brightness features extracted from different color spaces.

The project compared:

  * Threshold-based classification
  * Linear SVM classification

The features used were:

  * Mean brightness
  * Brightness standard deviation

| Color Space | Threshold Accuracy | SVM Accuracy |
|---|---:|---:|
| RGB | 0.42 | 0.77 |
| HSV | 0.65 | 0.82 |
| LAB | 0.71 | 0.85 |

LAB with SVM achieved the best result. This showed that even simple features can work well when the right color space and classifier are used.

### 8. Image Similarity Retrieval

The image similarity module retrieves the most visually similar images to a query image.

It combines three feature types:

  * HSV color histograms
  * Canny edge maps
  * ResNet50 deep features

Cosine similarity was used to compare feature vectors. A weighted score was then calculated using color, edge, and deep feature similarities.

Default combined weights:

| Feature Type | Weight |
|---|---:|
| Color histogram | 0.4 |
| Edge features | 0.1 |
| Deep features | 0.5 |

Deep features gave the most consistent results, while color and edge features added interpretability. The combined method gave the best balance because it captured both low-level and high-level visual similarity.

### 9. Image Quality Assessment

The image quality module checks whether an image may be difficult to use for downstream computer vision tasks.

It detects:

  * Blurriness using Laplacian variance
  * Underexposure and overexposure using mean brightness
  * Noise using Gaussian residual standard deviation
  * Problem regions using face detection and visual annotations

| Quality Issue | Method Used | Threshold |
|---|---|---:|
| Blur | Variance of Laplacian | Less than 180 |
| Underexposure | Mean grayscale brightness | Less than 70 |
| Overexposure | Mean grayscale brightness | More than 180 |
| Noise | Gaussian residual standard deviation | More than 12 |

The output image includes a black header bar with the detected quality warnings and visual annotations where relevant.

### 10. Image Annotation and Cropping

The project includes a simple coordinate-based annotation tool. Users upload an image, view it with axes, enter the crop coordinates, and save the selected region of interest.

This helps users focus the detection algorithms on the most relevant part of the image. It is especially useful for cluttered images where the background may cause false positives.

### 11. Integration and Interface Design

The final notebook integrates the main modules into a command-line menu.

Main menu options:

  1. Detect Landmark
  2. Face, Eyes and Smile Detection
  3. Time of Day
  4. Image Similarity Search
  5. Assess Image Quality
  6. Frontal Car Detection in Traffic
  7. Crop Image
  0. Exit

The interface works best when each task is run individually in Google Colab. Because Colab has limitations with repeated `input()` loops, the full menu may behave more smoothly in a local Python terminal.

* * *

## Repository Structure

```text
.
├── FINAL_435_PROJECT.ipynb
├── SIFT_+_BoVW_+_SVM_Classification.ipynb
├── GLD(v2).ipynb
├── Face_Feature_Detection_(Classification).ipynb
├── car_frontal_detection (2).ipynb
├── student2_time_of_day.ipynb
└── README.md
```

### Notebook Purpose

| Notebook | Purpose |
|---|---|
| `FINAL_435_PROJECT.ipynb` | Integrated notebook containing the main menu and core modules |
| `SIFT_+_BoVW_+_SVM_Classification.ipynb` | Landmark classification experiment using SIFT, BoVW, and SVM |
| `GLD(v2).ipynb` | Google Landmark Dataset v2 exploration and failed template retrieval attempt |
| `Face_Feature_Detection_(Classification).ipynb` | Face, eye, and smile detection with evaluation |
| `car_frontal_detection (2).ipynb` | Frontal car detection using Haar cascades |
| `student2_time_of_day.ipynb` | Day and night classification using thresholding and SVM |

* * *

## Tech Stack

  * Python
  * OpenCV
  * NumPy
  * Matplotlib
  * scikit-learn
  * TensorFlow Keras
  * ResNet50
  * PIL
  * imagehash
  * Google Colab

* * *

## Key Results

  * Face detection achieved 86% overall accuracy with perfect recall for the face class
  * Eye detection achieved 95% accuracy and was the strongest facial feature detector
  * Smile detection achieved only 54% accuracy because of false positives and expression sensitivity
  * Frontal car detection achieved 74% accuracy, but struggled with angled and partially visible vehicles
  * Time-of-day classification performed best with LAB color space and linear SVM, reaching 85% accuracy
  * SIFT with BoVW and SVM produced high training accuracy but failed to generalize reliably to new landmark images
  * Haar cascades were selected for final landmark detection because they produced interpretable bounding boxes and worked better for the controlled custom dataset
  * Image similarity retrieval worked best when deep ResNet50 features were combined with color and edge features
  * Image quality assessment successfully flagged blur, exposure, and noise using lightweight OpenCV metrics

* * *

## Limitations

The project has several important limitations:

  * The landmark dataset was small, with fewer than 100 positive samples per landmark after augmentation
  * Haar cascades were sensitive to angle, lighting, scale, and partial occlusion
  * The SIFT and BoVW pipeline overfit the training data and did not perform well on unseen images
  * The Google Landmark Dataset v2 attempt failed because template images could not be downloaded reliably
  * Smile and car detection produced false positives because Haar cascades are sensitive to background patterns and pose variation
  * Time-of-day classification struggled with borderline cases such as dusk, indoor lighting, and mixed illumination
  * Image quality assessment used fixed thresholds, which may not adapt well to every scene or camera type
  * Google Colab limited the interactivity of the menu and cropping workflow

* * *

## Future Work

Future improvements could include:

  * Expanding the landmark dataset with more diverse images and stronger annotations
  * Replacing Haar cascades with modern object detection models such as YOLO, Faster R-CNN, or SSD
  * Using CNN-based landmark classification instead of handcrafted SIFT and BoVW features
  * Improving smile detection with a facial expression recognition model
  * Training a custom vehicle detector for different angles and traffic conditions
  * Replacing fixed image quality thresholds with adaptive or learning-based methods
  * Building a web interface with Streamlit, Flask, or FastAPI instead of relying on Google Colab inputs
  * Adding automated evaluation scripts and cleaner dataset versioning
  * Improving the annotation tool with drag-and-drop bounding boxes instead of coordinate entry

* * *

## Getting Started

### Option 1: Run in Google Colab

Upload the notebooks to Google Colab and run the setup cells first.

Install the required packages if needed:

```bash
pip install opencv-python numpy matplotlib scikit-learn tensorflow pillow imagehash tqdm joblib
```

Upload the required images, trained cascade XML files, and dataset folders into the Colab runtime before running detection or retrieval tasks.

### Option 2: Run Locally

Clone the repository:

```bash
git clone https://github.com/ibrawr/<repository-name>.git
cd <repository-name>
```

Install the dependencies:

```bash
pip install opencv-python numpy matplotlib scikit-learn tensorflow pillow imagehash tqdm joblib
```

Open Jupyter Notebook:

```bash
jupyter notebook
```

Then run:

```text
FINAL_435_PROJECT.ipynb
```

The integrated notebook contains the main menu and connects the core project modules.

* * *

## Example Workflow

A typical workflow is:

  1. Upload an image into the runtime folder
  2. Run the import and setup cells
  3. Run the menu function in `FINAL_435_PROJECT.ipynb`
  4. Enter the image filename
  5. Select one of the available computer vision tasks
  6. Review the printed result and the annotated output image

Example menu:

```text
=== MAIN MENU ===
1. Detect Landmark
2. Face, Eyes & Smile Detection
3. Time of Day
4. Image Similarity Search
5. Assess Image Quality
6. Frontal Car Detection in Traffic
7. Crop Image
0. Exit
```

* * *

## Project Status

This project is complete as an academic computer vision system for CSCI435 Computer Vision Algorithms and Systems.

The final result is a modular OpenCV-based image analysis toolkit that demonstrates multiple classical computer vision techniques in one integrated workflow. Some modules achieved strong results, especially face detection, eye detection, and LAB-based time-of-day classification. Other modules, such as smile detection, car detection, and landmark recognition, revealed the limitations of small datasets and traditional Haar cascade methods.

The project is included as a portfolio project because it shows practical computer vision implementation, dataset preparation, model comparison, evaluation, system integration, and honest analysis of model limitations.
