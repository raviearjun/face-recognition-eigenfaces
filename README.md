# Face Detection & Recognition with Eigenfaces and SVM

## Overview

This project implements a face recognition system using Eigenfaces (PCA) and a Support Vector Machine (SVM) classifier. It covers:

- Loading and preprocessing images
- Detecting and cropping faces
- Feature extraction via PCA (Eigenfaces)
- Model training and evaluation
- Visualization of Eigenfaces and classification results
- Real-time face recognition from a webcam

## Directory Structure

```
├── images/                     # Dataset folder, subdirectories per person label
│   ├── person1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── person2/
│       ├── img1.jpg
│       └── img2.jpg
├── eigenface_pipeline.pkl      # Saved trained pipeline (PCA + SVM)
├── face_detection_recognition_tutorial.ipynb  # Jupyter notebook with code
└── README.md                   # Project description and usage
```

## Installation

1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```
2. Install dependencies (preferably in a virtual environment):
   ```bash
   pip install opencv-python numpy matplotlib scikit-learn
   ```

## Usage

1. **Prepare dataset**: Place your face images under `images/<label>/` directories.
2. **Run the notebook**: Open `face_detection_recognition_tutorial.ipynb` and execute cells to:
   - Load and preprocess images
   - Train the PCA+SVM pipeline
   - Evaluate on a test split (classification report)
   - Save the trained model to `eigenface_pipeline.pkl`
   - Visualize top Eigenfaces
3. **Real-time Recognition**:
   In a new cell (or separate script), use the following snippet to perform webcam-based recognition:
   ```python
   import cv2, numpy as np, pickle
   from your_module import detect_faces, crop_faces, resize_and_flatten, eigenface_prediction

   # Load trained pipeline
   with open('eigenface_pipeline.pkl', 'rb') as f:
       pipe = pickle.load(f)

   cap = cv2.VideoCapture(0)
   while True:
       ret, frame = cap.read()
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       try:
           scores, labels, faces = eigenface_prediction(gray)
           for (x, y, w, h), label, score in zip(faces, labels, scores):
               cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
               cv2.putText(frame, f"{label} ({score:.2f})", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
       except:
           cv2.putText(frame, "No face detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
       cv2.imshow('Eigenface Recognition', frame)
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
   cap.release()
   cv2.destroyAllWindows()
   ```

## Key Components

- **load_image**: Read and convert images to grayscale
- **detect_faces**: Haar Cascade-based face detection
- **crop_faces**: Crop and optionally select a single face
- **resize_and_flatten**: Resize to 128×128 and flatten for PCA
- **MeanCentering**: Custom transformer to center pixel intensities
- **PCA**: Extract top principal components (Eigenfaces)
- **SVC**: Linear SVM for classification
- **eigenface_prediction**: Wrapper to predict labels and scores for face crops

## Visualization

- **Eigenfaces**: Display the top `n_components` eigenvectors reshaped as face images
- **Classification Report**: Precision, recall, F1-score for each class

## Dependencies

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib
- Scikit-learn

## License

This project is free to use for educational purpose
