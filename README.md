# Potato Disease Classification App

This Streamlit application allows you to classify potato diseases using your device's camera. The app can identify Early Blight, Late Blight, and Healthy potato leaves.

## Features

- Real-time camera input for disease classification
- Support for three classes: Early Blight, Late Blight, and Healthy
- Confidence score display
- User-friendly interface with instructions
- Responsive design

## Prerequisites

- Python 3.8 or higher
- Webcam or camera-enabled device
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure you have a trained model saved as 'potato_disease_model.h5' in the root directory
2. Run the Streamlit app:
```bash
streamlit run app.py
```
3. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)
4. Allow camera access when prompted
5. Position the potato leaf in the camera view and take a picture
6. View the classification results

## Model Requirements

The application expects a trained TensorFlow model saved as 'potato_disease_model.h5' with the following specifications:
- Input image size: 256x256 pixels
- Output classes: 3 (Early Blight, Late Blight, Healthy)
- Model should be trained on RGB images

## Tips for Best Results

- Ensure good lighting when taking pictures
- Keep the camera steady
- Focus on the leaf area
- Avoid blurry images
- Make sure the leaf is clearly visible in the frame

## License

This project is licensed under the MIT License - see the LICENSE file for details. 